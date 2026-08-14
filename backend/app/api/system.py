"""파라미터 스키마 · GPU 조회 같은 시스템 정보 API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.services import gpu, models, param_schema

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/params/schema")
def params_schema() -> dict[str, Any]:
    """폼을 그리는 데 필요한 모든 정보. 프론트는 이걸 렌더링만 한다."""
    return {"schema": param_schema.build_schema(), "presets": param_schema.PRESETS}


@router.get("/system/gpus")
def system_gpus() -> dict[str, Any]:
    return {"gpus": gpu.list_gpus()}


@router.get("/system/weights")
def system_weights() -> dict[str, Any]:
    """모델 경로 입력창의 자동완성 후보. 고정 목록이 아니라 제안일 뿐이다."""
    return {"candidates": models.candidates()}


@router.post("/system/validate-model")
def validate_model(payload: dict[str, Any]) -> dict[str, Any]:
    """입력한 모델 경로를 즉시 판정한다. 학습 시작 전에 틀린 걸 알려주기 위한 것."""
    return models.resolve(str(payload.get("model", "")))


@router.get("/system/info")
def system_info() -> dict[str, Any]:
    """선택 기능의 사용 가능 여부. 프론트가 토글을 비활성화하는 근거로 쓴다."""

    def importable(module: str) -> bool:
        import importlib.util

        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    return {
        "tensorboard": importable("tensorboard"),
        "tensorrt": importable("tensorrt"),
        "onnx": importable("onnx"),
    }
