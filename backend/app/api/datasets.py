"""데이터셋 등록 API — zip 업로드와 로컬 경로 지정 두 경로."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core import db
from app.core.config import DATASETS_DIR, IMAGE_SUFFIXES, MAX_ZIP_BYTES, UPLOADS_DIR
from app.services import dataset_ingest

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
    return [db.row_to_dataset(r) for r in db.query("SELECT * FROM datasets ORDER BY created_at DESC")]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if row is None:
        raise HTTPException(404, "데이터셋을 찾을 수 없습니다.")
    return db.row_to_dataset(row)


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
                        "name": dataset["classes"][cid] if cid < len(dataset["classes"]) else str(cid),
                        "cx": cx, "cy": cy, "w": w, "h": h,
                    }
                )
        samples.append({"path": str(image), "boxes": boxes})
        if len(samples) >= limit:
            break
    return {"samples": samples}


@router.get("/{dataset_id}/image")
def dataset_image(dataset_id: str, path: str) -> FileResponse:
    """샘플 뷰어용 이미지 서빙. 등록된 데이터셋 루트 안쪽만 허용한다."""
    dataset = get_dataset(dataset_id)
    root = Path(dataset["root"]).resolve()
    target = Path(path).resolve()
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
        raise HTTPException(409, "이 데이터셋을 쓴 학습 기록이 있어 삭제할 수 없습니다.")
    # 경로 참조 데이터셋의 원본은 절대 지우지 않는다. 우리가 만든 폴더만 지운다.
    shutil.rmtree(DATASETS_DIR / dataset_id, ignore_errors=True)
    db.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    return {"status": "deleted", "id": dataset["id"]}
