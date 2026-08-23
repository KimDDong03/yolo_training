"""학습 파라미터 폼의 단일 진실 원천.

프론트는 여기서 내려주는 스키마를 렌더링만 한다. 입력 필드를 프론트에 하드코딩하지 않는다.
기본값은 ultralytics DEFAULT_CFG_DICT 에서 읽으므로 ultralytics 버전이 올라가도 자동으로 따라간다.
"""

from __future__ import annotations

import math
from typing import Any

from app.core.config import WEIGHTS_DIR

# ultralytics 가 배포하는 표준 사전학습 가중치. 단독망에서는 bundle/weights 에 실제 파일이 있어야 한다.
KNOWN_MODELS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

# (key, label, type, group, advanced, min, max, step, choices, help)
_SPEC: list[tuple[Any, ...]] = [
    # --- 기본 ---
    ("model", "모델", "path", "기본", False, None, None, None, None,
     "가중치(.pt) 또는 모델 정의(.yaml) 경로. 이전 학습의 best.pt 를 넣으면 이어서 학습한다."),
    ("epochs", "에폭 수", "int", "기본", False, 1, 10000, 1, None,
     "전체 데이터셋을 몇 번 반복할지."),
    ("time", "시간 예산(시간)", "float", "기본", False, 0.0, 24.0, 0.25, None,
     "0이면 끕니다. 켜면 에폭 수를 무시하고 이 시간만큼만 학습하며, 에폭 수는 예산에 맞춰 "
     "자동으로 다시 계산됩니다. 정확한 상한은 아닙니다 — 마지막 에폭의 검증과 마무리가 "
     "더 걸리므로 실제 소요는 예산을 조금 넘습니다. 조기 종료(patience)가 먼저 걸리면 "
     "그보다 일찍 끝납니다. 모자이크 종료 에폭(close_mosaic)과 함께 쓰면 에폭 수가 계속 "
     "바뀌는 탓에 그 시점이 걸리지 않을 수 있습니다."),
    ("imgsz", "이미지 크기", "int", "기본", False, 32, 4096, 32, None,
     "학습 입력 해상도(정사각형). 32의 배수여야 한다. 크면 작은 객체에 유리하지만 VRAM을 많이 쓴다."),
    ("batch", "배치 크기", "int", "기본", False, -1, 1024, 1, None,
     "-1이면 VRAM에 맞춰 자동 결정한다."),
    ("patience", "조기 종료 인내", "int", "기본", False, 0, 1000, 1, None,
     "이 에폭 수만큼 성능이 개선되지 않으면 학습을 멈춘다. 0이면 끄기."),
    ("seed", "랜덤 시드", "int", "기본", False, 0, 2**31 - 1, 1, None,
     "같은 시드 + deterministic 이면 결과가 재현된다."),
    ("pretrained", "사전학습 가중치 사용", "bool", "기본", True, None, None, None, None,
     "끄면 가중치를 처음부터(random init) 학습한다."),
    # --- 하드웨어 ---
    ("workers", "데이터 로더 워커", "int", "하드웨어", False, 0, 32, 1, None,
     "데이터 로딩 프로세스 수. Windows에서는 너무 크면 오히려 느려진다."),
    ("amp", "AMP(혼합정밀)", "bool", "하드웨어", False, None, None, None, None,
     "학습을 빠르게 하고 VRAM을 아낀다. 손실이 NaN이 되면 꺼본다."),
    ("cache", "이미지 캐시", "enum", "하드웨어", True, None, None, None, ["False", "ram", "disk"],
     "이미지를 메모리/디스크에 캐시해 로딩 병목을 줄인다. ram은 데이터셋이 작을 때만."),
    # --- 옵티마이저 ---
    ("optimizer", "옵티마이저", "enum", "옵티마이저", True, None, None, None,
     ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
     "auto면 데이터셋 크기에 따라 ultralytics가 고른다."),
    ("lr0", "초기 학습률", "float", "옵티마이저", True, 1e-6, 1.0, 1e-5, None,
     "너무 크면 발산하고 너무 작으면 수렴이 느리다."),
    ("lrf", "최종 학습률 비율", "float", "옵티마이저", True, 1e-5, 1.0, 1e-3, None,
     "학습 종료 시점의 학습률 = lr0 × lrf."),
    ("momentum", "모멘텀", "float", "옵티마이저", True, 0.0, 0.999, 0.001, None, ""),
    ("weight_decay", "가중치 감쇠", "float", "옵티마이저", True, 0.0, 0.1, 1e-5, None, ""),
    ("warmup_epochs", "웜업 에폭", "float", "옵티마이저", True, 0.0, 20.0, 0.1, None,
     "학습 초반에 학습률을 서서히 올리는 구간."),
    ("warmup_momentum", "웜업 모멘텀", "float", "옵티마이저", True, 0.0, 0.999, 0.01, None, ""),
    ("cos_lr", "코사인 스케줄러", "bool", "옵티마이저", True, None, None, None, None,
     "학습률을 코사인 곡선으로 낮춘다. 긴 학습에서 보통 유리하다."),
    # --- 손실 가중치 ---
    ("box", "박스 손실 가중치", "float", "손실 가중치", True, 0.0, 20.0, 0.1, None, ""),
    ("cls", "분류 손실 가중치", "float", "손실 가중치", True, 0.0, 20.0, 0.1, None, ""),
    ("dfl", "DFL 손실 가중치", "float", "손실 가중치", True, 0.0, 20.0, 0.1, None, ""),
    # --- 증강 ---
    ("hsv_h", "색조 변화", "float", "증강", True, 0.0, 1.0, 0.001, None, ""),
    ("hsv_s", "채도 변화", "float", "증강", True, 0.0, 1.0, 0.01, None, ""),
    ("hsv_v", "명도 변화", "float", "증강", True, 0.0, 1.0, 0.01, None, ""),
    ("degrees", "회전(도)", "float", "증강", True, -180.0, 180.0, 1.0, None, ""),
    ("translate", "이동", "float", "증강", True, 0.0, 1.0, 0.01, None, ""),
    ("scale", "확대/축소", "float", "증강", True, 0.0, 2.0, 0.01, None, ""),
    ("shear", "전단", "float", "증강", True, -180.0, 180.0, 1.0, None, ""),
    ("perspective", "원근", "float", "증강", True, 0.0, 0.001, 1e-5, None, ""),
    ("flipud", "상하 반전 확률", "float", "증강", True, 0.0, 1.0, 0.05, None, ""),
    ("fliplr", "좌우 반전 확률", "float", "증강", True, 0.0, 1.0, 0.05, None, ""),
    ("mosaic", "모자이크", "float", "증강", True, 0.0, 1.0, 0.05, None,
     "4장을 이어 붙여 한 장으로 학습한다. 작은 객체 검출에 크게 도움된다."),
    ("close_mosaic", "모자이크 종료 에폭", "int", "증강", True, 0, 100, 1, None,
     "마지막 N 에폭 동안 모자이크를 끈다. 실제 분포에 맞춰 마무리하는 효과."),
    ("mixup", "믹스업", "float", "증강", True, 0.0, 1.0, 0.05, None, ""),
    ("copy_paste", "카피-페이스트", "float", "증강", True, 0.0, 1.0, 0.05, None, ""),
    ("erasing", "랜덤 지우기", "float", "증강", True, 0.0, 0.9, 0.05, None, ""),
    # --- 학습 전략 ---
    ("rect", "직사각형 학습", "bool", "학습 전략", True, None, None, None, None,
     "패딩을 줄여 속도를 올린다. 배치 내 종횡비가 비슷할 때 유효."),
    ("multi_scale", "멀티 스케일", "bool", "학습 전략", True, None, None, None, None, ""),
    ("single_cls", "단일 클래스로 취급", "bool", "학습 전략", True, None, None, None, None,
     "모든 라벨을 한 클래스로 합쳐 학습한다."),
    ("dropout", "드롭아웃", "float", "학습 전략", True, 0.0, 0.9, 0.05, None, ""),
    ("fraction", "데이터 사용 비율", "float", "학습 전략", True, 0.01, 1.0, 0.01, None,
     "데이터셋의 일부만 써서 빠르게 실험할 때."),
    ("val", "에폭마다 검증", "bool", "학습 전략", True, None, None, None, None,
     "끄면 지표 그래프가 그려지지 않는다."),
    ("deterministic", "결정론적 실행", "bool", "학습 전략", True, None, None, None, None, ""),
    # --- 저장 ---
    ("save_period", "체크포인트 주기", "int", "저장", True, -1, 1000, 1, None,
     "N 에폭마다 가중치를 따로 저장한다. -1이면 저장하지 않는다."),
    ("plots", "플롯 생성", "bool", "저장", True, None, None, None, None,
     "끄면 예측 미리보기와 종료 후 플롯이 생성되지 않는다."),
]

# UI 전용 값 — ultralytics 학습 인자가 아니다. model.train(**params) 에 넘기면 TypeError 가 난다.
# (key, label, type, group, advanced, default, help)
_OPTION_SPEC: list[tuple[Any, ...]] = [
    ("tensorboard", "TensorBoard 로그", "bool", "저장", True, False,
     "켜면 ultralytics 가 TensorBoard 로그도 함께 남긴다. 별도로 tensorboard 를 실행해 봐야 한다."),
]

# 기본 화면에서 처음부터 보이는 순서
GROUP_ORDER = ["기본", "하드웨어", "옵티마이저", "손실 가중치", "증강", "학습 전략", "저장"]


def default_model() -> str:
    """폼에 처음 채워둘 모델. 번들에 있는 가장 작은 .pt 를 고르고, 없으면 내장 정의로 떨어진다."""
    if WEIGHTS_DIR.is_dir():
        preferred = [p for name in KNOWN_MODELS for p in (WEIGHTS_DIR / name,) if p.is_file()]
        if preferred:
            return str(preferred[0].resolve())
        any_pt = sorted(WEIGHTS_DIR.glob("*.pt"))
        if any_pt:
            return str(any_pt[0].resolve())
    return "yolo11n.yaml"


def _defaults() -> dict[str, Any]:
    from ultralytics.utils import DEFAULT_CFG_DICT

    return dict(DEFAULT_CFG_DICT)


def build_schema() -> dict[str, Any]:
    defaults = _defaults()

    fields: list[dict[str, Any]] = []
    for key, label, type_, group, advanced, min_, max_, step, choices, help_ in _SPEC:
        default = defaults.get(key)
        if key == "model":
            default = default_model()
        elif key == "cache":
            default = "False" if not default else str(default)
        elif key == "time":
            # ultralytics 기본값이 None 이다. 그대로 두면 폼이 null 을 렌더하고
            # 사용자가 손대지 않은 값이 그대로 돌아와 숫자 입력이 비어 보인다.
            default = 0.0 if default is None else float(default)

        # 기본값 보정과 별개로 돌아야 한다. elif 로 묶여 있던 동안 cache 만 choices 가
        # 원시 문자열 리스트로 남았고, _coerce 의 c["value"] 가 문자열을 인덱싱해
        # 학습 시작이 500 으로 죽었다.
        if choices is not None:
            choices = [{"value": c, "label": c, "available": True} for c in choices]

        # ultralytics 기본값의 타입이 우리 선언과 다를 수 있다 (multi_scale 은 bool 로 쓰지만
        # DEFAULT_CFG_DICT 에는 0.0 으로 들어 있다). 맞춰 두지 않으면 폼이 그 값을 그대로
        # 돌려보내고 _coerce 가 자기 기본값을 거부한다.
        if type_ == "int" and isinstance(default, float):
            default = int(default)
        elif type_ == "bool" and not isinstance(default, bool):
            default = bool(default)

        fields.append(
            {
                "key": key,
                "label": label,
                "type": type_,
                "group": group,
                "advanced": advanced,
                "default": default,
                "min": min_,
                "max": max_,
                "step": step,
                "choices": choices,
                "help": help_,
                "scope": "params",
            }
        )

    for key, label, type_, group, advanced, default, help_ in _OPTION_SPEC:
        fields.append(
            {
                "key": key,
                "label": label,
                "type": type_,
                "group": group,
                "advanced": advanced,
                "default": default,
                "min": None,
                "max": None,
                "step": None,
                "choices": None,
                "help": help_,
                "scope": "options",
            }
        )

    return {"groups": GROUP_ORDER, "fields": fields}


def field_index() -> dict[str, dict[str, Any]]:
    return {f["key"]: f for f in build_schema()["fields"]}


def defaults_dict(scope: str = "params") -> dict[str, Any]:
    return {f["key"]: f["default"] for f in build_schema()["fields"] if f["scope"] == scope}


class ValidationError(Exception):
    """폼 값 검증 실패. 메시지를 사용자에게 그대로 보여준다."""


def _coerce(field: dict[str, Any], value: Any) -> Any:
    key, type_ = field["key"], field["type"]
    if value is None:
        return field["default"]

    if type_ == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValidationError(f"{key}: 참/거짓 값이어야 합니다.")

    if type_ in {"int", "float"}:
        if isinstance(value, bool) or isinstance(value, (list, dict)):
            raise ValidationError(f"{key}: 숫자여야 합니다.")
        try:
            number = int(value) if type_ == "int" else float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{key}: 숫자로 해석할 수 없습니다 ({value!r}).") from None
        if not math.isfinite(number):
            raise ValidationError(f"{key}: 유한한 숫자여야 합니다.")
        if field["min"] is not None and number < field["min"]:
            raise ValidationError(f"{key}: {field['min']} 이상이어야 합니다 (입력 {number}).")
        if field["max"] is not None and number > field["max"]:
            raise ValidationError(f"{key}: {field['max']} 이하여야 합니다 (입력 {number}).")
        return number

    if type_ == "enum":
        allowed = {c["value"] for c in (field["choices"] or [])}
        text = str(value)
        if allowed and text not in allowed:
            raise ValidationError(f"{key}: 허용되지 않은 값입니다 ({text!r}).")
        return text

    return str(value)


def validate(values: dict[str, Any] | None, scope: str) -> dict[str, Any]:
    """폼에서 온 값을 스키마 allowlist 로 걸러 검증한다.

    이 함수가 유일한 관문이다. 없으면 임의의 키가 그대로 model.train(**params) 로 흘러들어가고,
    UI 전용 값(options)이 학습 인자에 섞여 TypeError 를 낸다.
    """
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValidationError(f"{scope} 는 객체여야 합니다.")

    index = field_index()
    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        field = index.get(key)
        if field is None:
            raise ValidationError(f"알 수 없는 항목입니다: {key}")
        if field["scope"] != scope:
            raise ValidationError(f"{key} 는 {scope} 가 아니라 {field['scope']} 에 속합니다.")
        cleaned[key] = _coerce(field, value)
    return cleaned


PRESETS: dict[str, dict[str, Any]] = {
    "빠른 테스트": {"epochs": 10, "imgsz": 640, "batch": -1, "patience": 0},
    "균형": {"epochs": 100, "imgsz": 640, "batch": -1, "patience": 30},
    "고정밀": {"epochs": 300, "imgsz": 800, "batch": -1, "patience": 50, "cos_lr": True, "close_mosaic": 20},
}
