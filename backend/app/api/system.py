"""파라미터 스키마 · GPU 조회 같은 시스템 정보 API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.services import gpu, param_schema

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/params/schema")
def params_schema() -> dict[str, Any]:
    """폼을 그리는 데 필요한 모든 정보. 프론트는 이걸 렌더링만 한다."""
    return {"schema": param_schema.build_schema(), "presets": param_schema.PRESETS}


@router.get("/system/gpus")
def system_gpus() -> dict[str, Any]:
    return {"gpus": gpu.list_gpus()}
