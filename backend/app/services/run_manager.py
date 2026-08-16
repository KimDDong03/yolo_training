"""실행 큐 · GPU 슬롯 할당 · 워커 프로세스 기동/정지.

GPU 1장이면 자연스럽게 순차 실행이 되고, 장수가 늘면 같은 코드가 병렬 실행이 된다.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.core import db, procs
from app.core.config import BACKEND_DIR, HOOKS_DIR, RUNS_DIR, WEIGHTS_DIR, offline_env
from app.services import gpu, jobs, models, param_schema

POLL_INTERVAL = 1.0

# config.json 이 allowlist 검증을 거쳤음을 나타낸다. 이 값이 다르면 기동 직전에 다시 거른다.
CONFIG_SCHEMA_VERSION = 2
# 폼이 아니라 서버가 채우는 학습 인자.
SERVER_OWNED_PARAMS = {"device"}

# run_id -> Popen (이 백엔드가 직접 띄운 것)
_processes: dict[str, subprocess.Popen] = {}
# run_id -> pid (백엔드 재시작 후 살아 있는 걸 발견해 다시 붙잡은 것)
_adopted: dict[str, int] = {}
# 큐 선점은 API 스레드와 스케줄러 루프가 동시에 시도할 수 있다.
# 사이드잡 기동(start_job)과 run/dataset 삭제도 같은 락을 쓴다 — 셋이 GPU 슬롯과
# 소유자 폴더를 놓고 겹치기 때문이다.
_schedule_lock = threading.RLock()


# PID 생존·신원 확인은 app.core.procs 로 옮겼다 — 학습 워커와 사이드잡이 같은 문제를 쓴다.
# 기존 호출부를 그대로 두려고 이름만 붙여 둔다.
_pid_alive = procs.pid_alive
_pid_is_our_worker = procs.is_our_worker


class RunError(Exception):
    """사용자에게 그대로 보여줄 수 있는 실행 실패."""


def run_dir_for(run_id: str) -> Path:
    return RUNS_DIR / run_id


def training_devices() -> set[int]:
    used: set[int] = set()
    for row in db.query("SELECT devices FROM runs WHERE status = 'running'"):
        used.update(json.loads(row["devices"]))
    return used


def busy_devices() -> set[int]:
    """지금 GPU 를 붙잡고 있는 것 전부.

    사이드잡(내보내기·분석)도 포함해야 한다 — 프로세스를 분리해도 VRAM 은 분리되지
    않으므로, 학습과 동시에 돌면 둘 다 OOM 난다.
    """
    return training_devices() | jobs.reserved_devices()


def create_run(
    name: str,
    dataset: dict[str, Any],
    params: dict[str, Any],
    options: dict[str, Any],
    devices: list[int],
    retry_of: str | None = None,
) -> dict[str, Any]:
    merged = param_schema.defaults_dict("params")
    merged.update(params)
    merged.pop("name", None)
    if merged.get("cache") in ("False", "false", False, None):
        merged["cache"] = False
    merged["device"] = ",".join(str(d) for d in devices) if devices else "cpu"

    # 모델 참조를 절대 경로로 확정한다. 워커의 cwd 가 달라서 상대 경로는 여기서만 풀린다.
    # 폴더를 만들기 전에 해석한다 — 실패했을 때 빈 run 폴더가 남지 않도록.
    resolved_model = models.require(str(merged.get("model", "")))

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    run_dir = run_dir_for(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 여기부터 실패하면 반쯤 만들어진 run 폴더가 남는다 → 되돌린다.
    try:
        model_path = Path(resolved_model)
        if model_path.is_file() and model_path.suffix.lower() == ".pt":
            # 큐에서 대기하는 동안 원본(다른 run 의 best.pt 등)이 사라져도 안전하도록 복사해 둔다.
            # 몇 MB~수십 MB 라 비용이 거의 없고, "이 run 이 무슨 가중치로 시작했는지"가 run 폴더에 남는다.
            inputs = run_dir / "inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            copied = inputs / model_path.name
            shutil.copyfile(model_path, copied)
            merged["model"] = str(copied.resolve())
        else:
            merged["model"] = resolved_model

        merged_options = param_schema.defaults_dict("options")
        merged_options.update(options)

        config = {
            "run_id": run_id,
            "dataset_id": dataset["id"],
            "data": dataset["yaml_path"],
            "params": merged,
            "options": merged_options,
            "devices": devices,
            "source_model": resolved_model,  # 사용자가 고른 원본 (params.model 은 run 폴더 안의 복사본)
            "retry_of": retry_of,
            "schema_version": CONFIG_SCHEMA_VERSION,
            "created_at": time.time(),
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        db.execute(
            "INSERT INTO runs"
            " (id, name, dataset_id, status, params, options, devices, retry_of, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, name, dataset["id"], "queued", json.dumps(merged, default=str),
             json.dumps(merged_options, default=str), json.dumps(devices), retry_of, time.time()),
        )
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    row = db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        # 바로 위에서 INSERT 했으니 도달하지 않는다. 도달했다면 DB 가 이상한 것이므로
        # None 을 안쪽까지 흘려보내 정체불명 에러를 만들지 말고 여기서 말한다.
        raise RunError("실행을 만들었지만 다시 읽지 못했습니다.")
    return db.row_to_run(row)


def _preflight(run_id: str) -> None:
    """기동 직전 재검증. 큐에서 대기하는 동안 파일이 사라졌을 수 있다."""
    run_dir = run_dir_for(run_id)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    data_yaml = Path(config["data"])
    if not data_yaml.is_file():
        raise RunError(f"데이터셋 정의를 찾을 수 없습니다: {data_yaml}")

    models.require(str(config["params"].get("model", "")))

    # 이 버전 이전에 큐에 들어간 run 은 allowlist 를 거치지 않은 config 를 갖고 있다.
    # 워커는 params 를 그대로 model.train(**params) 로 넘기므로, 모르는 키가 있으면 TypeError 가 난다.
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        allowed = set(param_schema.field_index()) | SERVER_OWNED_PARAMS
        dropped = sorted(k for k in config["params"] if k not in allowed)
        if dropped:
            _append_event(run_id, {
                "t": "warning",
                "message": f"알 수 없는 학습 인자를 무시했습니다: {', '.join(dropped)}",
            })
            for key in dropped:
                config["params"].pop(key, None)
        config["schema_version"] = CONFIG_SCHEMA_VERSION
        config.setdefault("options", {})
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _spawn(run_id: str, started_at: float) -> None:
    _preflight(run_id)
    run_dir = run_dir_for(run_id)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    options = config.get("options") or {}

    env = os.environ.copy()
    env.update(offline_env())
    env["YOLOWEB_RUN_DIR"] = str(run_dir)
    # UI 옵션은 환경변수로 넘긴다. config.json 은 워커 본문에서 읽는데,
    # DDP 자식은 sitecustomize 단계에서 이미 ultralytics 를 import 하므로 그때는 늦다.
    env["YOLOWEB_TENSORBOARD"] = "1" if options.get("tensorboard") else "0"
    # DDP 자식 프로세스까지 sitecustomize 를 태우기 위한 경로 주입
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HOOKS_DIR), str(BACKEND_DIR), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["PYTHONUNBUFFERED"] = "1"

    log_file = open(run_dir / "train.log", "ab", buffering=0)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    # 번들 가중치 폴더를 작업 디렉터리로 삼으면 오프라인에서도 사전학습 가중치와
    # AMP 체크용 가중치를 다운로드 없이 찾는다.
    cwd = WEIGHTS_DIR if WEIGHTS_DIR.is_dir() else BACKEND_DIR

    process = subprocess.Popen(
        [sys.executable, str(BACKEND_DIR / "train_worker.py"), "--run-dir", str(run_dir)],
        cwd=str(cwd),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    _processes[run_id] = process
    db.execute("UPDATE runs SET pid = ? WHERE id = ?", (process.pid, run_id))
    # started_at 은 claim 시점에 이미 기록했다. PID 신원 확인의 기준 시각이라 여기서 덮지 않는다.
    _ = started_at


def _finish(run_id: str, status: str, error: str | None = None) -> None:
    db.execute(
        "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
        (status, error, time.time(), run_id),
    )
    _processes.pop(run_id, None)
    _adopted.pop(run_id, None)


def _last_event(run_id: str) -> dict[str, Any] | None:
    path = run_dir_for(run_id) / "events.jsonl"
    if not path.exists():
        return None
    last = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _append_event(run_id: str, payload: dict[str, Any]) -> None:
    payload.setdefault("ts", time.time())
    path = run_dir_for(run_id) / "events.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _settle(run_id: str, code: int | None) -> None:
    """종료된 워커의 최종 상태를 events.jsonl 기준으로 확정한다."""
    last = _last_event(run_id)
    if last and last.get("t") == "end":
        _finish(run_id, last.get("status", "completed"), last.get("error"))
        return
    # 워커가 end 이벤트를 남기지 못하고 죽었다 → 백엔드가 대신 남긴다.
    # 프론트는 "end 가 올 때까지 진행 중" 규칙만 지키면 되므로 이 보정이 필요하다.
    stopped = (run_dir_for(run_id) / "stop.flag").exists()
    status = "stopped" if stopped else "failed"
    error = None if stopped else f"학습 프로세스가 코드 {code} 로 종료되었습니다. train.log 를 확인하세요."
    _append_event(run_id, {"t": "end", "status": status, "error": error})
    _finish(run_id, status, error)


def reap() -> None:
    """끝난 워커를 정리하고 DB 상태를 맞춘다."""
    for run_id, process in list(_processes.items()):
        code = process.poll()
        if code is not None:
            _settle(run_id, code)
    for run_id, pid in list(_adopted.items()):
        if not _pid_alive(pid):
            _settle(run_id, None)
    # 끝난 사이드잡의 GPU 예약을 푼다. 아무도 상태를 조회하지 않아도 슬롯이 풀려야 한다.
    jobs.sweep()


def recover() -> None:
    """백엔드 재시작 시 running 으로 남아 있던 run 을 정리하거나 다시 붙잡는다.

    워커는 백엔드의 자식이지만 Windows 에서 부모가 죽어도 함께 죽지 않는다.
    그래서 아직 살아 있으면 상태를 유지하고 PID 로 다시 추적한다 —
    백엔드를 재시작해도 학습이 이어지고, 브라우저는 events.jsonl 을 다시 읽어 복원된다.
    """
    for row in db.query("SELECT * FROM runs WHERE status = 'running'"):
        run_id = row["id"]
        if run_id in _processes or run_id in _adopted:
            continue
        last = _last_event(run_id)
        if last and last.get("t") == "end":
            _finish(run_id, last.get("status", "completed"), last.get("error"))
            continue
        pid = row["pid"]
        if pid and _pid_is_our_worker(int(pid), row["started_at"]):
            _adopted[run_id] = int(pid)
            continue
        error = "백엔드가 재시작되었고 학습 프로세스도 살아 있지 않습니다."
        _append_event(run_id, {"t": "end", "status": "failed", "error": error})
        _finish(run_id, "failed", error)


def schedule() -> None:
    """여유 GPU 슬롯이 있으면 큐에서 꺼내 기동한다.

    API 핸들러(스레드풀)와 스케줄러 루프가 동시에 이 함수를 부를 수 있으므로,
    락 안에서 조건부 UPDATE 로 큐를 선점한다. 선점에 성공한 쪽만 프로세스를 띄운다.
    """
    with _schedule_lock:
        queued = db.query("SELECT * FROM runs WHERE status = 'queued' ORDER BY created_at")
        if not queued:
            return
        busy = busy_devices()
        cpu_running = any(
            not json.loads(row["devices"])
            for row in db.query("SELECT devices FROM runs WHERE status = 'running'")
        )
        for row in queued:
            devices = json.loads(row["devices"])
            if devices:
                if busy & set(devices):
                    continue
            elif cpu_running:
                continue

            started_at = time.time()
            claimed = db.execute(
                "UPDATE runs SET status = 'running', started_at = ? WHERE id = ? AND status = 'queued'",
                (started_at, row["id"]),
            )
            if claimed.rowcount != 1:
                continue  # 다른 쪽이 먼저 가져갔거나 그 사이 정지됐다

            if devices:
                busy.update(devices)
            else:
                cpu_running = True
            try:
                _spawn(row["id"], started_at)
            except Exception as exc:  # noqa: BLE001 - 선점만 하고 못 띄우면 상태를 되돌린다
                error = f"학습 프로세스를 시작하지 못했습니다: {exc}"
                _append_event(row["id"], {"t": "end", "status": "failed", "error": error})
                _finish(row["id"], "failed", error)


def stop_run(run_id: str, mode: str = "graceful") -> None:
    run_dir = run_dir_for(run_id)
    if not run_dir.is_dir():
        raise RunError("실행을 찾을 수 없습니다.")

    with _schedule_lock:
        row = db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if row is None:
            raise RunError("실행을 찾을 수 없습니다.")
        if row["status"] == "queued":
            # 대기 취소도 end 이벤트를 남긴다. 프론트는 "end 가 올 때까지 진행 중"이라는
            # 단일 규칙만 지키면 되고, 스트림 리더도 이걸 보고 정리된다.
            _append_event(run_id, {"t": "end", "status": "stopped"})
            _finish(run_id, "stopped")
            return
        if row["status"] != "running":
            raise RunError("이미 종료된 실행입니다.")

    (run_dir / "stop.flag").write_text("stop", encoding="utf-8")
    if mode != "force":
        return

    process = _processes.get(run_id)
    if process is not None:
        # 자식(DDP rank·데이터로더 워커)까지 함께 정리한다.
        procs.kill_tree(process.pid)
        return

    # 재시작 후 다시 붙잡은 run 은 PID 밖에 없다. 신원이 확인될 때만 죽인다.
    # PID 는 재사용되므로 이 검증 없이 kill 하면 무관한 프로세스를 죽일 수 있다.
    pid = row["pid"]
    if not _pid_is_our_worker(pid, row["started_at"]):
        raise RunError(
            "학습 프로세스를 확인할 수 없어 강제 종료하지 않았습니다. 안전 정지를 쓰거나 상태를 확인하세요."
        )
    procs.kill_tree(int(pid))


# --------------------------------------------------------------------- 사이드잡

@contextlib.contextmanager
def exclusive_delete(owner_type: str, owner_id: str):
    """소유자를 지우는 동안 사이드잡이 기동되지 못하게 막는다.

    "잡이 도는지 확인하고 → 지운다" 를 따로 하면 그 사이에 잡이 시작될 수 있다. 그러면
    실행 중인 워커의 폴더를 지우게 된다. start_job 과 같은 락을 잡아야 실제로 배타가 된다.
    삭제는 드물고 스케줄러는 어차피 1초 주기라, 파일 삭제 동안 락을 쥐는 비용은 감수한다.
    """
    with _schedule_lock:
        live = jobs.live_for(owner_type, owner_id)
        if live:
            labels = ", ".join(sorted({jobs.spec_for(j["kind"]).label for j in live}))
            raise RunError(f"{labels} 작업이 진행 중입니다. 끝난 뒤에 삭제하세요.")
        yield


def start_job(
    kind: str,
    owner_type: str,
    owner_id: str,
    args: dict[str, Any],
    preferred: list[int] | None = None,
) -> dict[str, Any]:
    """사이드잡을 기동한다. GPU 슬롯 배정은 여기서만 한다.

    jobs 모듈이 스스로 판단하지 않는 이유: "지금 GPU 가 비었나" 는 학습 큐를 아는 이 모듈만
    답할 수 있고, 이 모듈은 이미 jobs 를 쓴다. 양쪽이 서로를 import 하면 순환이 된다.
    학습 기동과 같은 락 안에서 배정해야 둘이 같은 GPU 를 동시에 집지 않는다.
    """
    spec = jobs.spec_for(kind)
    clean = spec.validate(args)
    with _schedule_lock:
        devices: list[int] = []
        if spec.needs_gpu(clean):
            busy = busy_devices()
            wanted: list[int] = (
                preferred
                if preferred is not None
                else [int(g["index"]) for g in gpu.list_gpus()]  # type: ignore[arg-type]
            )
            free = [d for d in wanted if d not in busy]
            if not free and not spec.gpu_optional:
                raise RunError(
                    "이 작업은 GPU 가 필요한데 지금 다른 작업이 쓰고 있습니다. "
                    "끝난 뒤 다시 시도하세요."
                )
            # GPU 를 못 잡아도 되는 작업은 CPU 로 내려서 돌린다. 거절하는 것보다 낫고,
            # 무엇보다 학습이 쓰는 GPU 를 뺏지 않는다.
            # 몇 장을 잡을지는 스펙이 정한다(기본 1). 요청보다 여유가 적으면 있는 만큼만 잡는다 —
            # 거절하면 두 장짜리 요청이 한 장 남은 기계에서 영영 시작되지 않는다.
            devices = free[: max(spec.devices_wanted(clean), 1)]
        try:
            return jobs.spawn(kind, owner_type, owner_id, clean, devices)
        except sqlite3.IntegrityError as exc:
            # 부분 유니크 인덱스 위반 = 같은 소유자에 같은 종류의 잡이 이미 돌고 있다.
            # 안 잡으면 친절한 안내가 500 으로 퇴화한다.
            raise RunError(f"이미 {spec.label} 작업이 진행 중입니다.") from exc


async def scheduler_loop() -> None:
    while True:
        try:
            reap()
            schedule()
            # 순환 import 회피: anomaly 는 run_manager 를 쓰므로 여기서 늦게 가져온다.
            from app.services import anomaly

            anomaly.scan()
        except Exception:  # 스케줄러는 어떤 경우에도 죽으면 안 된다
            import traceback

            traceback.print_exc()
        await asyncio.sleep(POLL_INTERVAL)
