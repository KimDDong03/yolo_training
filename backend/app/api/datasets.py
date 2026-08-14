"""데이터셋 등록 API — zip 업로드와 로컬 경로 지정 두 경로."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core import db, fsops
from app.core.config import DATASETS_DIR, IMAGE_SUFFIXES, MAX_ZIP_BYTES, UPLOADS_DIR
from app.services import dataset_ingest, param_schema, recommend, run_manager

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _persist(meta: dict[str, Any]) -> dict[str, Any]:
    db.execute(
        "INSERT INTO datasets (id, name, source, origin, root, yaml_path, classes, report, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (
            meta["id"],
            meta["name"],
            meta["source"],
            meta["origin"],
            meta["root"],
            meta["yaml_path"],
            json.dumps(meta["classes"], ensure_ascii=False),
            json.dumps(meta["report"], ensure_ascii=False),
            meta["created_at"],
        ),
    )
    return meta


@router.get("")
def list_datasets() -> list[dict[str, Any]]:
    return [
        db.row_to_dataset(r)
        for r in db.query("SELECT * FROM datasets ORDER BY created_at DESC")
    ]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if row is None:
        raise HTTPException(404, "데이터셋을 찾을 수 없습니다.")
    return db.row_to_dataset(row)


@router.post("/{dataset_id}/recommendation")
def dataset_recommendation(dataset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """이 데이터셋에 맞는 파라미터 제안. 부작용 없는 계산이다.

    지금 폼에 들어 있는 값을 함께 받아야 "무엇을 무엇으로 바꾸라" 를 말할 수 있어서
    GET 이 아니라 POST 다 (validate-model 과 같은 성격).
    """
    row = db.query_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if row is None:
        raise HTTPException(404, "데이터셋을 찾을 수 없습니다.")

    # 계산만 하는 엔드포인트라도 allowlist 를 거친다. 여기서 나온 patch 가 그대로 폼에
    # 들어갔다가 학습 인자가 되기 때문이다.
    try:
        params = param_schema.validate(payload.get("params"), "params")
    except param_schema.ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    devices = payload.get("devices") or []
    if not isinstance(devices, list) or any(not isinstance(d, int) for d in devices):
        raise HTTPException(422, "devices 는 GPU 번호의 배열이어야 합니다.")

    return recommend.recommend(db.row_to_dataset(row), params, devices)


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(""),
    val_ratio: float = Form(0.2),
) -> dict[str, Any]:
    """zip 을 받아 해제하고 등록한다."""
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "zip 파일만 업로드할 수 있습니다.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS_DIR / f"{uuid.uuid4().hex}.zip"
    size = 0
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ZIP_BYTES:
                    raise HTTPException(413, "zip 파일이 상한을 넘습니다.")
                out.write(chunk)

        meta = dataset_ingest.ingest_zip(
            tmp, name or Path(file.filename or "dataset").stem, val_ratio=val_ratio
        )
    except dataset_ingest.IngestError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)

    return _persist(meta)


@router.post("/path")
def register_path(payload: dict[str, Any]) -> dict[str, Any]:
    """서버 로컬 폴더를 복사 없이 참조 등록한다."""
    path = str(payload.get("path", "")).strip()
    if not path:
        raise HTTPException(400, "폴더 경로가 비어 있습니다.")
    try:
        meta = dataset_ingest.ingest_path(
            path,
            str(payload.get("name") or Path(path).name),
            val_ratio=float(payload.get("val_ratio", 0.2)),
        )
    except dataset_ingest.IngestError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _persist(meta)


@router.get("/{dataset_id}/samples")
def dataset_samples(dataset_id: str, limit: int = 12) -> dict[str, Any]:
    """검수용 샘플 이미지 목록과 각 이미지의 GT 박스."""
    dataset = get_dataset(dataset_id)
    list_file = DATASETS_DIR / dataset_id / "train.txt"
    if not list_file.exists():
        return {"samples": []}

    samples: list[dict[str, Any]] = []
    for line in list_file.read_text(encoding="utf-8").splitlines()[: limit * 3]:
        image = Path(line.strip())
        if not image.exists() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = dataset_ingest._label_for(image)
        boxes: list[dict[str, Any]] = []
        if label.exists():
            for row in label.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = row.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    continue
                boxes.append(
                    {
                        "cls": cid,
                        "name": dataset["classes"][cid]
                        if cid < len(dataset["classes"])
                        else str(cid),
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                    }
                )
        samples.append({"path": str(image), "boxes": boxes})
        if len(samples) >= limit:
            break
    return {"samples": samples}


@router.get("/{dataset_id}/review")
def dataset_review(
    dataset_id: str, category: str = "", offset: int = 0, limit: int = 24
) -> dict[str, Any]:
    """문제 유형별 이미지 목록.

    숫자만 보여주면 판단에 못 쓴다 — 실제로 어떤 이미지인지 열어봐야 라벨링 실수인지 배경인지 안다.
    """
    dataset = get_dataset(dataset_id)
    try:
        review = dataset_ingest.load_review(
            DATASETS_DIR / dataset_id, Path(dataset["root"])
        )
    except dataset_ingest.IngestError as exc:
        raise HTTPException(400, str(exc)) from exc

    categories = review.get("categories", {})
    summary = [
        {k: v for k, v in entry.items() if k != "items"}
        for entry in categories.values()
    ]
    summary.sort(key=lambda c: c["total"], reverse=True)

    page: dict[str, Any] = {"items": [], "total": 0, "stored": 0, "truncated": False}
    if category:
        entry = categories.get(category)
        if entry is None:
            raise HTTPException(404, f"알 수 없는 검수 항목입니다: {category}")
        items = entry.get("items", [])
        page = {
            "items": items[offset : offset + limit],
            "total": entry["total"],
            "stored": entry["stored"],
            "truncated": entry["truncated"],
        }

    return {
        "categories": summary,
        "box_stats": review.get("box_stats", {}),
        "review_cap": review.get("review_cap", 0),
        "category": category,
        "offset": offset,
        "limit": limit,
        **{"page": page},
    }


@router.get("/{dataset_id}/image")
def dataset_image(dataset_id: str, path: str) -> FileResponse:
    """샘플 뷰어용 이미지 서빙. 등록된 데이터셋 루트 안쪽만 허용한다."""
    dataset = get_dataset(dataset_id)
    root = Path(dataset["root"]).resolve()
    # 검수 목록은 root 기준 상대 경로를 주고, 샘플 목록은 절대 경로를 준다 — 둘 다 받는다.
    raw = Path(path)
    target = (raw if raw.is_absolute() else root / raw).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(403, "데이터셋 폴더 밖의 파일은 열 수 없습니다.")
    if not target.is_file() or target.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(404, "이미지를 찾을 수 없습니다.")
    return FileResponse(target)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, str]:
    dataset = get_dataset(dataset_id)
    used = db.query_one("SELECT id FROM runs WHERE dataset_id = ?", (dataset_id,))
    if used is not None:
        raise HTTPException(
            409, "이 데이터셋을 쓴 학습 기록이 있어 삭제할 수 없습니다."
        )
    # 경로 참조 데이터셋의 원본은 절대 지우지 않는다. 우리가 만든 폴더만 지운다.
    # 파일을 먼저 지우고 성공했을 때만 DB 행을 지운다 — 실패를 삼키면 목록에서만
    # 사라지고 디스크에는 사본이 남아 손댈 방법이 없어진다.
    # 데이터셋에 붙는 잡(임베딩 등)이 도는 중이면 Windows 파일 잠금으로 삭제가 실패하는데,
    # 그때 나오는 메시지로는 사용자가 원인을 알 수 없다. 잡 검사를 먼저 한다.
    try:
        with run_manager.exclusive_delete("dataset", dataset_id):
            fsops.remove_tree(DATASETS_DIR / dataset_id)
    except run_manager.RunError as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            409,
            "데이터셋 폴더를 지우지 못했습니다. 파일을 열어 둔 프로그램이 있는지"
            f" 확인하고 다시 시도하세요: {exc}",
        ) from exc

    db.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    return {"status": "deleted", "id": dataset["id"]}
