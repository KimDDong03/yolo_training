"""모델(가중치) 참조를 해석·검증하는 단일 지점.

경로 해석이 여기 한 곳에 모여 있어야 하는 이유:
워커의 작업 디렉터리는 오프라인 가중치 탐색 때문에 `bundle/weights` 다(run_manager._spawn).
그래서 `storage/runs/.../best.pt` 같은 상대 경로는 **API 프로세스에서는 존재하고 워커에서는 존재하지 않는다.**
후보 목록 · 즉시 검증 · run 생성 · 기동 직전 재검증이 전부 이 모듈을 통해야 그 함정을 피한다.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import BASE_DIR, RUNS_DIR, WEIGHTS_DIR
from app.services import run_summary

WEIGHT_SUFFIXES = {".pt"}
CONFIG_SUFFIXES = {".yaml", ".yml"}
# 자동완성에 노출할 내장 정의: 기본 검출 모델(yolo11n.yaml, yolov8s.yaml …)만.
_SUGGESTED_CONFIG = re.compile(r"^yolo(?:v)?\d+[nsmlx]?\.yaml$")


class ModelError(Exception):
    """사용자에게 그대로 보여줄 수 있는 모델 참조 오류."""


def _ultralytics_cfg_dir() -> Path | None:
    try:
        import ultralytics

        path = Path(ultralytics.__file__).parent / "cfg" / "models"
        return path if path.is_dir() else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def builtin_configs() -> tuple[str, ...]:
    """ultralytics 내장 모델 정의(.yaml). 사전학습 없이 처음부터 학습할 때 쓴다.

    ultralytics 는 `yolo11.yaml` 하나로 n/s/m/l/x 를 처리하고 파일명의 스케일 문자로 크기를 고른다.
    그래서 파일명만 모으면 실제로 쓸 수 있는 `yolo11n.yaml` 이 목록에서 빠진다 — scales 키를 펼쳐야 한다.
    """
    cfg_dir = _ultralytics_cfg_dir()
    if cfg_dir is None:
        return ()

    import yaml

    names: set[str] = set()
    for path in cfg_dir.rglob("*.yaml"):
        names.add(path.name)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scales = (data or {}).get("scales") if isinstance(data, dict) else None
        if isinstance(scales, dict):
            for scale in scales:
                names.add(f"{path.stem}{scale}{path.suffix}")
    return tuple(sorted(names))


def _is_builtin_config(name: str) -> bool:
    """ultralytics 가 실제로 로드할 수 있는 이름인지 그쪽 로더로 판정한다."""
    if name in builtin_configs():
        return True
    try:
        from ultralytics.nn.tasks import yaml_model_load

        yaml_model_load(name)
        return True
    except Exception:
        return False


def resolve(reference: str) -> dict[str, Any]:
    """모델 참조를 해석한다. 예외를 던지지 않고 판정 결과를 돌려준다.

    Returns:
        {ok, kind: 'weights'|'config'|'unknown', resolved, message}
    """
    text = str(reference or "").strip().strip('"')
    if not text:
        return {"ok": False, "kind": "unknown", "resolved": None, "message": "모델 경로가 비어 있습니다."}

    path = Path(text)
    suffix = path.suffix.lower()

    # 1) 내장 모델 정의 (.yaml) — ultralytics 가 이름만으로 찾는다. 파일 경로일 수도 있다.
    if suffix in CONFIG_SUFFIXES:
        for candidate in _candidate_paths(path):
            if candidate.is_file():
                return {"ok": True, "kind": "config", "resolved": str(candidate.resolve()),
                        "message": "모델 정의 파일 (처음부터 학습)"}
        # 디렉터리가 붙은 경로는 그 파일이 실제로 있어야 한다.
        # 파일명만 떼어 내장 정의와 비교하면 `C:\오타\yolo11n.yaml` 이 조용히 내장 yolo11n 으로 바뀐다.
        if path.parent == Path(".") and _is_builtin_config(path.name):
            return {"ok": True, "kind": "config", "resolved": path.name,
                    "message": "ultralytics 내장 모델 정의 (처음부터 학습)"}
        return {"ok": False, "kind": "config", "resolved": None,
                "message": f"내장 모델 정의에도 없고 파일도 아닙니다: {text}"}

    if suffix not in WEIGHT_SUFFIXES:
        return {"ok": False, "kind": "unknown", "resolved": None,
                "message": "가중치(.pt) 또는 모델 정의(.yaml) 경로여야 합니다."}

    # 2) 절대/상대 경로로 실제 존재하는 .pt
    for candidate in _candidate_paths(path):
        if candidate.is_file():
            return {"ok": True, "kind": "weights", "resolved": str(candidate.resolve()),
                    "message": f"가중치 파일 ({_human_size(candidate)})"}

    # 3) 경로가 아닌 순수 모델명 — ultralytics 가 인터넷에서 받아오려 한다
    if path.parent == Path("."):
        return {
            "ok": False,
            "kind": "weights",
            "resolved": None,
            "message": (
                f"'{text}' 은(는) 파일이 아니라 모델 이름입니다. "
                "단독망에서는 자동 다운로드가 실패하므로 bundle/weights 에 파일을 넣고 경로로 지정하세요."
            ),
        }

    return {"ok": False, "kind": "weights", "resolved": None, "message": f"파일을 찾을 수 없습니다: {text}"}


def _candidate_paths(path: Path) -> list[Path]:
    if path.is_absolute():
        return [path]
    # 상대 경로는 프로젝트 루트와 번들 가중치 폴더를 기준으로 본다.
    return [BASE_DIR / path, WEIGHTS_DIR / path, path]


def _human_size(path: Path) -> str:
    try:
        mb = path.stat().st_size / 1024**2
    except OSError:
        return "크기 확인 불가"
    return f"{mb:.1f} MB"


def require(reference: str) -> str:
    """해석에 성공하면 해석된 값을, 실패하면 ModelError 를 던진다."""
    result = resolve(reference)
    if not result["ok"]:
        raise ModelError(result["message"])
    return str(result["resolved"])


def _best_metric(run_dir: Path) -> float | None:
    """events.jsonl 의 epoch 이벤트에서 최고 mAP50-95 를 뽑는다.

    사이드바 요약과 같은 계산이라 run_summary 가 한 벌만 든다. 캐시도 거기 붙어 있다.
    """
    return run_summary.summarize(run_dir)["best_map"]


def candidates(limit_runs: int = 30) -> list[dict[str, Any]]:
    """자동완성 후보. 고정 목록이 아니라 '제안'이다 — 사용자는 아무 경로나 직접 넣을 수 있다."""
    items: list[dict[str, Any]] = []

    if WEIGHTS_DIR.is_dir():
        for path in sorted(WEIGHTS_DIR.glob("*.pt")):
            items.append({
                "value": str(path.resolve()),
                "label": path.name,
                "detail": f"번들 가중치 · {_human_size(path)}",
                "kind": "weights",
            })

    if RUNS_DIR.is_dir():
        run_dirs = sorted(
            (p for p in RUNS_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )[:limit_runs]
        for run_dir in run_dirs:
            best = run_dir / "train" / "weights" / "best.pt"
            if not best.is_file():
                continue
            metric = _best_metric(run_dir)
            detail = f"이전 학습 결과 · {run_dir.name}"
            if metric is not None:
                detail += f" · 최고 mAP50-95 {metric:.4f}"
            items.append({
                "value": str(best.resolve()),
                "label": f"{run_dir.name} / best.pt",
                "detail": detail,
                "kind": "weights",
            })

    # 내장 정의는 290개쯤 된다. 전부 제안하면 자동완성이 쓸모없어지므로 기본 검출 모델만 추린다.
    # (제안일 뿐이라 사용자는 seg/pose 등 다른 이름도 직접 입력할 수 있고, resolve() 가 그대로 받아준다.)
    for name in builtin_configs():
        if _SUGGESTED_CONFIG.match(name):
            items.append({
                "value": name,
                "label": name,
                "detail": "내장 모델 정의 · 사전학습 없이 처음부터",
                "kind": "config",
            })

    return items
