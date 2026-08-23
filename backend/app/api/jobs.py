"""사이드잡(내보내기·분석 등)의 공통 API.

잡 종류가 늘어도 이 파일은 그대로다. 새 종류는 services/jobs.py 의 SPECS 에 등록하면
여기 엔드포인트를 자동으로 얻는다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core import db
from app.services import jobs, run_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# 소유자 종류마다 존재 확인 방법이 다르다. 여기 없는 종류는 아예 받지 않는다.
_OWNER_TABLES = {"run": "runs", "dataset": "datasets"}


def _owner_or_404(owner_type: str, owner_id: str) -> None:
    table = _OWNER_TABLES.get(owner_type)
    if table is None:
        raise HTTPException(404, f"알 수 없는 소유자 종류입니다: {owner_type}")
    if db.query_one(f"SELECT id FROM {table} WHERE id = ?", (owner_id,)) is None:
        raise HTTPException(404, "대상을 찾을 수 없습니다.")


def _spec_or_404(kind: str):
    try:
        return jobs.spec_for(kind)
    except jobs.JobError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{owner_type}/{owner_id}")
def list_jobs(owner_type: str, owner_id: str) -> dict[str, Any]:
    _owner_or_404(owner_type, owner_id)
    kinds = [k for k, s in jobs.SPECS.items() if s.owner_type == owner_type]
    return {"jobs": [jobs.status(k, owner_type, owner_id) for k in kinds]}


@router.get("/{owner_type}/{owner_id}/{kind}")
def job_status(owner_type: str, owner_id: str, kind: str) -> dict[str, Any]:
    _owner_or_404(owner_type, owner_id)
    _spec_or_404(kind)
    return jobs.status(kind, owner_type, owner_id)


@router.post("/{owner_type}/{owner_id}/{kind}")
def start_job(
    owner_type: str, owner_id: str, kind: str, payload: dict[str, Any]
) -> dict[str, Any]:
    _owner_or_404(owner_type, owner_id)
    _spec_or_404(kind)
    try:
        return run_manager.start_job(kind, owner_type, owner_id, payload)
    except jobs.JobError as exc:
        raise HTTPException(422, str(exc)) from exc
    except run_manager.RunError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/{owner_type}/{owner_id}/{kind}/stop")
def stop_job(owner_type: str, owner_id: str, kind: str) -> dict[str, Any]:
    _owner_or_404(owner_type, owner_id)
    _spec_or_404(kind)
    jobs.stop(kind, owner_type, owner_id)
    return jobs.status(kind, owner_type, owner_id)


@router.get("/{owner_type}/{owner_id}/{kind}/files/{path:path}")
def job_file(owner_type: str, owner_id: str, kind: str, path: str) -> FileResponse:
    """잡 산출물 서빙.

    경계를 명시적으로 좁힌다 — 소유자 종류는 allowlist 로 받고, 뿌리는 범위는
    <소유자>/jobs/<kind>/ 하나뿐이며, 실제 경로를 resolve 한 뒤 그 안쪽인지 확인한다.
    이 검사가 없으면 path 에 ../ 를 넣어 디스크의 아무 파일이나 읽어갈 수 있다.
    """
    _owner_or_404(owner_type, owner_id)
    _spec_or_404(kind)
    root = jobs.job_dir(kind, owner_type, owner_id).resolve()
    target = (root / path).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    # 같은 URL 의 내용이 바뀐다 — 다시 검사하면 quality.json 이 통째로 새 파일이 된다.
    # FileResponse 는 Cache-Control 을 붙이지 않아서 브라우저가 Last-Modified 로 유효기간을
    # 스스로 정하고 재확인 없이 옛 본문을 내준다. 실제로 사진을 지운 뒤 화면이 지워진 사진을
    # 계속 보여줬다.
    return FileResponse(target, headers={"Cache-Control": "no-store"})
