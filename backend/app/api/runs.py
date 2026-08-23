"""학습 실행 API + 실시간 스트림 WebSocket."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse

from app.core import db, fsops
from app.services import (
    dataset_ingest,
    diagnose_fail,
    event_stream,
    gpu,
    jobs,
    label_issues,
    models,
    next_actions,
    param_schema,
    predict,
    run_manager,
    run_summary,
)

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _run_or_404(run_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        raise HTTPException(404, "실행을 찾을 수 없습니다.")
    return db.row_to_run(row)


@router.get("")
def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row in db.query("SELECT * FROM runs ORDER BY created_at DESC"):
        run = db.row_to_run(row)
        # 사이드바가 목록에서 바로 진행률과 최고 mAP 를 보여준다. DB 에 없는 값이라
        # events.jsonl 에서 뽑는데, 2초 폴링이라 run_summary 가 (mtime, size) 로 캐시한다.
        run["summary"] = run_summary.summarize(run_manager.run_dir_for(run["id"]))
        runs.append(run)
    return runs


def _source_model(run_id: str) -> str | None:
    """사용자가 고른 원본 가중치 경로. 못 읽으면 None.

    `params["model"]` 은 run 폴더 안의 복사본이라 run 마다 다르다
    (run_manager.py:105 — 큐에서 대기하는 동안 원본이 사라져도 안전하도록 복사한다).
    그래서 같은 가중치로 시작한 실행끼리도 params 만 보면 서로 달라 보인다.
    비교 화면이 "무엇이 달랐나" 를 물을 때 그 차이는 잡음이다.

    파일 이름으로 비교하는 것으로는 부족하다 — 서로 다른 실행에서 이어받은 best.pt 두 개는
    이름이 같지만 실제로 다른 가중치다. 원본 경로가 있어야 그 둘을 가른다.
    """
    try:
        text = (run_manager.run_dir_for(run_id) / "config.json").read_text(encoding="utf-8")
        value = json.loads(text).get("source_model")
    except (OSError, ValueError):
        return None
    return value if isinstance(value, str) else None


@router.get("/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = _run_or_404(run_id)
    run["source_model"] = _source_model(run_id)
    dataset = db.query_one("SELECT * FROM datasets WHERE id = ?", (run["dataset_id"],))
    if dataset is None:
        run["dataset"] = None
    else:
        info = db.row_to_dataset(dataset)
        # 진단 화면의 갤러리가 이 데이터셋의 사진을 연다. 경로가 낡았으면 사진이 전부
        # 깨지는데, 사유가 없으면 "모델이 다 놓쳤다" 로 읽힌다.
        info["path_status"] = dataset_ingest.path_status(info)
        run["dataset"] = info
    return run


class _Common:
    """생성 계열 엔드포인트가 공유하는 검증 결과.

    단일 생성과 스윕이 같은 계약을 두 벌로 갖지 않게 한 곳에서 만든다.
    effective 를 함께 내는 이유는 `params` 가 폼이 보낸 키만 담기 때문이다 —
    기본값 병합은 원래 run_manager.create_run 이 하므로, 그 전에 model 같은 값을
    보려는 호출자(스윕)는 여기서 병합된 것을 받아야 KeyError 를 만나지 않는다.
    """

    def __init__(
        self,
        dataset: dict[str, Any],
        params: dict[str, Any],
        options: dict[str, Any],
        devices: list[int],
        effective: dict[str, Any],
    ) -> None:
        self.dataset = dataset
        self.params = params
        self.options = options
        self.devices = devices
        self.effective = effective


def _validate_common(payload: dict[str, Any]) -> _Common:
    dataset_row = db.query_one(
        "SELECT * FROM datasets WHERE id = ?", (payload.get("dataset_id"),)
    )
    if dataset_row is None:
        raise HTTPException(422, "데이터셋을 찾을 수 없습니다.")

    # 스키마 allowlist 로 걸러낸다. 이게 없으면 임의 키가 그대로 model.train(**params) 로 들어간다.
    try:
        params = param_schema.validate(payload.get("params"), "params")
        options = param_schema.validate(payload.get("options"), "options")
    except param_schema.ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw_devices = payload.get("devices")
    if raw_devices is None:
        raw_devices = [g["index"] for g in gpu.list_gpus()][:1]
    if not isinstance(raw_devices, list):
        raise HTTPException(422, "devices 는 GPU 번호의 배열이어야 합니다.")
    # 검사하면서 바로 담는다. 따로 확인하고 나중에 변환하면 아래로 넘어가는 값의 타입이
    # 서명(create_run 의 list[int])과 어긋난 채로 남는다.
    devices: list[int] = []
    for d in raw_devices:
        if not isinstance(d, int):
            raise HTTPException(422, "devices 는 GPU 번호의 배열이어야 합니다.")
        devices.append(d)
    if len(devices) != len(set(devices)):
        # device="0,0" 을 넘기면 ultralytics 가 같은 GPU 를 두 장으로 알고 DDP 를 시도한다.
        raise HTTPException(422, "같은 GPU 번호를 두 번 지정할 수 없습니다.")
    known = {g["index"] for g in gpu.list_gpus()}
    unknown = [d for d in devices if d not in known]
    if unknown:
        raise HTTPException(422, f"존재하지 않는 GPU 번호입니다: {unknown}")

    effective = {**param_schema.defaults_dict("params"), **params}
    return _Common(db.row_to_dataset(dataset_row), params, options, devices, effective)


# 스윕이 만들 수 있는 run 수의 상한.
#
# 6 인 이유는 임의가 아니다 — 프론트의 COMPARE_LIMIT(Sidebar.tsx)와 같은 값이다.
# 스윕의 결과는 비교 화면에 통째로 들어가야 의미가 있고, 비교 화면은 run 마다 상세와
# 전체 이벤트를 병렬로 받으므로 그 상한이 곧 이 상한이다. 한쪽만 바꾸면 스윕 직후
# 비교가 잘린다.
SWEEP_MAX = 6


def _sweep_axis(payload: dict[str, Any], effective: dict[str, Any]) -> dict[str, Any]:
    """훑을 축 하나를 고르고 그것이 훑을 수 있는 축인지 본다."""
    axis = payload.get("axis")
    if not isinstance(axis, str):
        raise HTTPException(422, "axis 는 파라미터 이름이어야 합니다.")
    field = param_schema.field_index().get(axis)
    if field is None or field["scope"] != "params":
        raise HTTPException(422, f"훑을 수 없는 항목입니다: {axis}")
    if field["type"] == "bool":
        # 값이 참/거짓 둘뿐이라 "여러 값을 훑는다" 가 성립하지 않는다.
        raise HTTPException(422, f"{field['label']} 은 참/거짓이라 훑을 수 없습니다.")
    if axis == "epochs" and float(effective.get("time") or 0) > 0:
        # 시간 예산이 켜져 있으면 ultralytics 가 매 에폭 끝에서 에폭 수를 예산에 맞춰
        # 다시 계산한다(engine/trainer.py:546). 그래서 서로 다른 에폭 수로 넣어도
        # 전부 같은 실행이 된다. 다르게 돌 수 없는 것은 실험이 아니다.
        raise HTTPException(
            422,
            "시간 예산이 켜져 있으면 에폭 수를 훑을 수 없습니다. "
            "예산이 에폭 수를 덮어써서 모든 실행이 같아집니다.",
        )
    return field


@router.post("/sweep")
def create_sweep(payload: dict[str, Any]) -> dict[str, Any]:
    """한 축을 여러 값으로 바꾼 run 을 한 번에 큐에 넣는다.

    **검증을 전부 끝낸 뒤에 만든다.** 클라이언트가 POST /api/runs 를 N번 부르면
    네 번째 값이 범위를 벗어날 때 앞의 세 개가 이미 만들어져 절반짜리 스윕이 남는다.

    보장하는 것은 "요청 검증이 실패하면 run 이 0개" 까지다. 만드는 도중 디스크가 차는
    것까지 되돌리지는 않는다 — create_run 이 폴더·복사·DB 를 개별로 커밋하고,
    스케줄러가 1초마다 독립적으로 돌아 뒤를 만드는 동안 앞이 이미 시작될 수 있다.
    """
    common = _validate_common(payload)
    field = _sweep_axis(payload, common.effective)
    axis = str(payload.get("axis"))

    values = payload.get("values")
    if not isinstance(values, list):
        raise HTTPException(422, "values 는 배열이어야 합니다.")
    if not 2 <= len(values) <= SWEEP_MAX:
        raise HTTPException(422, f"훑을 값은 2개 이상 {SWEEP_MAX}개 이하여야 합니다.")

    # 값마다 완전한 설정을 만들어 둔다. 여기서 하나라도 걸리면 아무것도 만들지 않는다.
    configs: list[tuple[Any, dict[str, Any]]] = []
    for value in values:
        try:
            cfg = param_schema.validate({**common.effective, axis: value}, "params")
        except param_schema.ValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        configs.append((cfg[axis], cfg))

    seen = [c[0] for c in configs]
    if len(seen) != len(set(seen)):
        # 320 과 "320" 은 검증을 지나면 같은 값이 된다(_coerce 가 축 타입으로 바꾼다).
        # 그래서 여기서는 문자열이 아니라 값 자체로 비교한다 — str() 로 비교하면
        # -0.0 과 0.0 이 서로 다른 문자열이라 같은 실행을 두 번 큐에 넣는다.
        raise HTTPException(422, "같은 값을 두 번 훑을 수 없습니다.")

    # 모델은 값마다 다를 수 있다(axis 가 model 인 스윕). create_run 안에서 늦게
    # 터지면 앞의 run 이 남으므로 여기서 전부 먼저 해석한다.
    for _, cfg in configs:
        try:
            models.require(str(cfg.get("model", "")))
        except models.ModelError as exc:
            raise HTTPException(422, str(exc)) from exc

    dataset = common.dataset
    base = str(payload.get("name") or dataset["name"])
    runs: list[dict[str, Any]] = []
    for value, cfg in configs:
        # 이름에 축과 값을 박는다. 사이드바에서 같은 접두어로 모여 보이고,
        # 비교 화면의 "다른 설정만" 표와 눈으로 대조할 수 있다.
        name = f"{base}/{axis}={value}"
        try:
            runs.append(
                run_manager.create_run(name, dataset, cfg, common.options, common.devices)
            )
        except models.ModelError as exc:  # 위에서 걸렀지만 파일이 그 사이 사라질 수 있다
            raise HTTPException(422, str(exc)) from exc
    run_manager.schedule()
    return {"runs": runs, "axis": axis, "label": field["label"]}


@router.post("")
def create_run(payload: dict[str, Any]) -> dict[str, Any]:
    common = _validate_common(payload)
    params, options, devices = common.params, common.options, common.devices

    retry_of = payload.get("retry_of")
    if retry_of is not None:
        if not isinstance(retry_of, str):
            raise HTTPException(422, "retry_of 는 실행 ID 여야 합니다.")
        if db.query_one("SELECT id FROM runs WHERE id = ?", (retry_of,)) is None:
            raise HTTPException(422, "재시도할 원본 실행을 찾을 수 없습니다.")

    dataset = common.dataset
    name = str(payload.get("name") or f"{dataset['name']}")
    try:
        run = run_manager.create_run(name, dataset, params, options, devices, retry_of)
    except models.ModelError as exc:
        raise HTTPException(422, str(exc)) from exc
    run_manager.schedule()
    return run


@router.get("/{run_id}/diagnosis")
def run_diagnosis(run_id: str) -> dict[str, Any]:
    """실패 원인과 처방, 그리고 고친 파라미터로 다시 돌릴 준비물.

    저장하지 않고 매번 계산한다 — 규칙을 개선하면 과거 실패에도 바로 적용된다.
    """
    _run_or_404(run_id)
    try:
        return diagnose_fail.diagnose(run_id)
    except run_manager.RunError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{run_id}/analysis/report")
def analysis_report(run_id: str) -> dict[str, Any]:
    """진단 리포트 전문. 분석 잡의 산출물이라 잡이 끝나야 생긴다."""
    _run_or_404(run_id)
    path = jobs.job_dir("analyze", "run", run_id) / "report.json"
    if not path.is_file():
        raise HTTPException(404, "아직 진단 결과가 없습니다.")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "진단 결과를 읽지 못했습니다.") from exc
    # 처방은 리포트에 굳히지 않고 여기서 얹는다 — 문구를 고치면 과거 리포트에도 바로 적용된다.
    report["next_actions"] = next_actions.build(report, _data_quality(report))
    # 라벨 오류 후보의 안내 문구도 같은 이유로 굳히지 않는다. 저장된 리포트에는 그 리포트를
    # 만들 때의 문장이 들어 있어, 문구를 고쳐도 재분석 전까지는 옛 문장이 뜬다.
    issues = report.get("label_issues")
    if isinstance(issues, dict) and issues.get("scope_note"):
        issues["scope_note"] = label_issues.SCOPE_NOTE
        _drop_retired_kinds(issues)
    return report


def _drop_retired_kinds(issues: dict[str, Any]) -> None:
    """접은 종류를 저장된 리포트에서도 걷어낸다 — scope_note 와 같은 이유다.

    사용자는 재분석 없이 예전 리포트를 열어 본다. 그냥 두면 접기로 한 바로 그 후보가
    계속 뜬다(`phantom_label` 이 이 경우였다). 이름을 박지 않고 `LABELS` 에 없는 종류를
    걷어내므로, 앞으로 다른 종류를 접어도 그대로 동작한다.

    `kinds[].count` 는 상한을 적용하기 **전** 건수라 `total` 에서 그만큼 빼면 정확하다.
    `next_actions` 가 이 `total` 을 읽으므로(next_actions.py) 같이 줄여야 두 화면이
    같은 숫자를 말한다.
    """
    live = label_issues.LABELS
    retired = [k for k in issues.get("kinds") or [] if k.get("kind") not in live]
    if not retired:
        return
    issues["kinds"] = [k for k in issues.get("kinds") or [] if k.get("kind") in live]
    issues["total"] = max(
        int(issues.get("total") or 0) - sum(int(k.get("count") or 0) for k in retired), 0
    )

    items = []
    for item in issues.get("items") or []:
        kept = [f for f in item.get("findings") or [] if f.get("kind") in live]
        # 남은 발견이 없으면 사진만 덩그러니 남으므로 목록에서 뺀다.
        if kept:
            item["findings"] = kept
            items.append(item)
    issues["items"] = items
    issues["shown"] = sum(len(item["findings"]) for item in items)


def _data_quality(report: dict[str, Any]) -> dict[str, Any] | None:
    """이 run 이 쓴 데이터셋의 품질 검사 결과. 안 돌렸으면 None.

    train/val 누수는 mAP 를 통째로 부풀리므로, 그 사실을 아는 화면은 mAP 를 보여 주는
    이 화면이어야 한다. 파일 읽기는 여기서 하고 next_actions.build 는 순수하게 남긴다.
    """
    dataset_id = report.get("dataset_id")
    if not dataset_id:
        return None
    try:
        path = jobs.job_dir("quality", "dataset", str(dataset_id)) / "quality.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (jobs.JobError, OSError, json.JSONDecodeError):
        # 품질 결과를 못 읽는다고 진단 리포트가 안 뜨면 안 된다.
        return None


@router.post("/{run_id}/stop")
def stop_run(run_id: str, mode: str = "graceful") -> dict[str, Any]:
    _run_or_404(run_id)
    try:
        run_manager.stop_run(run_id, mode)
    except run_manager.RunError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _run_or_404(run_id)


@router.delete("/{run_id}")
def delete_run(run_id: str) -> dict[str, str]:
    run = _run_or_404(run_id)
    if run["status"] in {"running", "queued"}:
        raise HTTPException(
            409, "진행 중인 학습은 삭제할 수 없습니다. 먼저 정지하세요."
        )
    # 파일을 먼저 지우고, 성공했을 때만 DB 행을 지운다. 순서를 바꾸거나 실패를 삼키면
    # 목록에서는 사라졌는데 디스크에는 가중치가 남아 손댈 방법이 없어진다.
    # 잡 검사와 삭제는 같은 락 안에서 해야 한다 — 따로 하면 그 사이에 잡이 시작된다.
    try:
        with run_manager.exclusive_delete("run", run_id):
            fsops.remove_tree(run_manager.run_dir_for(run_id))
    except run_manager.RunError as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            409,
            "산출물 폴더를 지우지 못했습니다. 파일을 열어 둔 프로그램이 있는지"
            f" 확인하고 다시 시도하세요: {exc}",
        ) from exc

    db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    return {"status": "deleted", "id": run_id}


@router.get("/{run_id}/events")
def run_events(run_id: str) -> dict[str, Any]:
    """WebSocket 없이도 전체 이벤트를 받아갈 수 있는 폴백."""
    _run_or_404(run_id)
    path = run_manager.run_dir_for(run_id) / "events.jsonl"
    events: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(event_stream.json_safe(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return {"events": events}


@router.get("/{run_id}/files/{path:path}")
def run_file(run_id: str, path: str) -> FileResponse:
    """run 디렉터리 안의 산출물(예측 이미지·플롯·가중치) 서빙."""
    _run_or_404(run_id)
    root = run_manager.run_dir_for(run_id).resolve()
    target = (root / path).resolve()
    if root not in target.parents:
        raise HTTPException(403, "실행 폴더 밖의 파일은 열 수 없습니다.")
    if not target.is_file():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    return FileResponse(
        target, filename=target.name if target.suffix == ".pt" else None
    )


@router.post("/{run_id}/predict")
async def run_predict(
    run_id: str,
    file: UploadFile = File(...),
    weights: str = Form("train/weights/best.pt"),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    imgsz: int = Form(640),
) -> dict[str, Any]:
    """학습된 가중치로 올린 이미지에 추론한다. 항상 CPU 로 돈다(predict.py 주석 참고)."""
    _run_or_404(run_id)
    root = run_manager.run_dir_for(run_id).resolve()
    target = (root / weights).resolve()
    if root not in target.parents:
        raise HTTPException(403, "실행 폴더 밖의 가중치는 쓸 수 없습니다.")

    payload = await file.read(predict.MAX_UPLOAD_BYTES + 1)
    try:
        return predict.run(root, target, payload, conf=conf, iou=iou, imgsz=imgsz)
    except predict.PredictError as exc:
        raise HTTPException(400, str(exc)) from exc


def _as_export(job: dict[str, Any]) -> dict[str, Any]:
    """잡 상태를 예전 내보내기 응답 모양으로 맞춘다.

    프론트(types.ts 의 ExportStatus)가 쓰는 키는 status/events/result/format 넷이고,
    TS 인터페이스는 여분의 키를 무시한다. 그래서 잡으로 갈아끼우면서 화면은 건드리지 않는다.
    """
    events = job.get("events") or []
    fmt = job.get("args", {}).get("format") or (
        events[0].get("format") if events else None
    )
    return {**job, "format": fmt}


@router.post("/{run_id}/export")
def start_export(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """가중치를 다른 포맷으로 변환한다. 오래 걸리므로 별도 프로세스로 띄우고 폴링으로 확인한다."""
    _run_or_404(run_id)
    weights = str(payload.get("weights", "train/weights/best.pt"))
    if not (run_manager.run_dir_for(run_id) / weights).is_file():
        raise HTTPException(404, f"가중치를 찾을 수 없습니다: {weights}")
    try:
        # GPU 가 필요한 포맷(TensorRT)은 이 run 이 쓰던 GPU 를 먼저 노린다.
        row = db.query_one("SELECT devices FROM runs WHERE id = ?", (run_id,))
        preferred = json.loads(row["devices"]) if row else None
        job = run_manager.start_job(
            "export", "run", run_id, {**payload, "weights": weights}, preferred
        )
    except jobs.JobError as exc:
        raise HTTPException(422, str(exc)) from exc
    except run_manager.RunError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _as_export(job)


@router.get("/{run_id}/export")
def export_status(run_id: str) -> dict[str, Any]:
    _run_or_404(run_id)
    return _as_export(jobs.status("export", "run", run_id))


@router.get("/{run_id}/weights")
def run_weights(run_id: str) -> dict[str, Any]:
    """추론에 쓸 수 있는 이 run 의 가중치 목록."""
    _run_or_404(run_id)
    root = run_manager.run_dir_for(run_id)
    items: list[dict[str, Any]] = []
    for path in sorted((root / "train" / "weights").glob("*.pt")):
        items.append(
            {
                "value": path.relative_to(root).as_posix(),
                "label": path.stem,
                "size_mb": round(path.stat().st_size / 1024**2, 1),
            }
        )
    return {"weights": items}


@router.get("/{run_id}/artifacts")
def run_artifacts(run_id: str) -> dict[str, Any]:
    """종료 후 플롯·가중치 목록. end 이벤트를 놓쳤을 때도 화면이 채워지도록 직접 스캔한다."""
    _run_or_404(run_id)
    root = run_manager.run_dir_for(run_id)
    save_dir = root / "train"
    plots: list[str] = []
    weights: list[str] = []
    if save_dir.is_dir():
        for file in sorted(save_dir.glob("*.png")) + sorted(save_dir.glob("*.jpg")):
            if file.name.startswith(("val_batch", "train_batch")):
                continue
            plots.append(file.relative_to(root).as_posix())
        for file in sorted((save_dir / "weights").glob("*.pt")):
            weights.append(file.relative_to(root).as_posix())

    epochs: dict[str, list[str]] = {}
    epochs_dir = root / "epochs"
    if epochs_dir.is_dir():
        for folder in sorted(
            epochs_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0
        ):
            if folder.is_dir():
                epochs[folder.name] = [
                    f.relative_to(root).as_posix() for f in sorted(folder.glob("*.jpg"))
                ]
    return {"plots": plots, "weights": weights, "epochs": epochs}


@router.websocket("/{run_id}/ws")
async def run_ws(websocket: WebSocket, run_id: str) -> None:
    row = db.query_one("SELECT id FROM runs WHERE id = ?", (run_id,))
    if row is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    stream = event_stream.manager.get(run_id, run_manager.run_dir_for(run_id))
    snapshot, queue = stream.subscribe()
    try:
        await websocket.send_json(snapshot)
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
        pass
    finally:
        stream.unsubscribe(queue)
        await event_stream.manager.release(run_id)
