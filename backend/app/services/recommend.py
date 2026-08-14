"""데이터셋의 성격을 보고 학습 파라미터를 제안한다.

등록 단계에서 이미 박스 크기·종횡비 분포를 계산해 두었는데(dataset_ingest._box_stats),
지금은 화면이 "작은 객체가 많습니다" 경고 한 줄로만 쓰고 버린다. 정작 그래서 imgsz 를
얼마로 해야 하는지는 사용자가 알아서 정해야 한다. 그 간극을 메운다.

param_schema 안에 넣지 않는 이유: 그쪽은 "이 폼에 무엇이 있고 무엇이 유효한가"(스키마)이고
여기는 "이 데이터에는 무엇이 좋은가"(정책)다. 섞으면 build_schema() 가 데이터셋을 알아야 한다.

불변식: 여기서 나가는 값은 반드시 param_schema.validate 를 통과한다. 통과하지 못하는 값을
제안하면 사용자가 '적용' 을 누른 뒤 학습 시작이 422 로 죽는다.
"""

from __future__ import annotations

from typing import Any

from app.services import estimate, param_schema

# imgsz 상한. 이보다 키우면 VRAM 이 급격히 늘어 대부분의 PC 에서 감당이 안 된다.
MAX_IMGSZ = 1280
# 작은 객체가 많을 때 목표로 삼는 해상도. {아주 많지는 않음: 960, 절반 이상: 1280}
TINY_TARGET_IMGSZ = {False: 960, True: 1280}


def round32(value: float) -> int:
    """imgsz 는 32의 배수여야 한다.

    param_schema._coerce 는 min/max 만 보고 step 을 검사하지 않는다. 700 같은 값을 제안하면
    ultralytics 가 조용히 736 으로 올려 시간·VRAM 예측이 어긋난다.
    """
    return max(32, int(round(value / 32)) * 32)


def _tiny_ratio(box_stats: dict[str, Any]) -> float:
    return float(box_stats.get("tiny_ratio") or 0.0)


def _imbalance(class_instances: dict[str, int]) -> tuple[float, str | None, str | None]:
    counts = {k: v for k, v in class_instances.items() if isinstance(v, int)}
    if len(counts) < 2:
        return 1.0, None, None
    most = max(counts, key=lambda k: counts[k])
    least = min(counts, key=lambda k: counts[k])
    if counts[least] <= 0:
        return float("inf"), most, least
    return counts[most] / counts[least], most, least


def recommend(
    dataset: dict[str, Any],
    params: dict[str, Any],
    devices: list[int],
) -> dict[str, Any]:
    report = dataset.get("report") or {}
    box_stats = report.get("box_stats")

    if not isinstance(box_stats, dict) or not box_stats.get("count"):
        # 이 컬럼이 생기기 전에 등록된 데이터셋이다. 여기서 다시 스캔하면 안 된다 —
        # 전체 라벨을 다시 읽는 수 분짜리 작업이 폼에 값을 입력하는 도중에 돈다.
        return {
            "available": False,
            "reason": "이 데이터셋은 박스 분포 정보 없이 등록되었습니다. 다시 등록하면 추천을 받을 수 있습니다.",
            "items": [],
            "advisories": [],
            "patch": {},
        }

    train_count = int(report.get("train_count") or report.get("total_images") or 0)
    tiny = _tiny_ratio(box_stats)
    median_area = float(box_stats.get("median_area") or 0.0)
    imgsz = int(params.get("imgsz", 640) or 640)
    epochs = int(params.get("epochs", 100) or 100)

    items: list[dict[str, Any]] = []
    advisories: list[dict[str, Any]] = []

    def add(
        rule: str, severity: str, changes: dict[str, Any], reason: str, effect: str
    ) -> None:
        # 지금 값과 같은 것은 제안하지 않는다.
        actual = {k: v for k, v in changes.items() if params.get(k) != v}
        if actual:
            items.append(
                {
                    "rule": rule,
                    "severity": severity,
                    "changes": {
                        k: {"from": params.get(k), "to": v} for k, v in actual.items()
                    },
                    "reason": reason,
                    "effect": effect,
                }
            )

    # --- 박스 크기 ---
    if tiny > 0.30:
        # 지금 값의 배수가 아니라 데이터가 요구하는 절대 목표를 제안한다.
        # 상대 증가로 하면 적용할 때마다 또 더 큰 값을 제안해 영영 수렴하지 않는다.
        target = min(round32(TINY_TARGET_IMGSZ[tiny > 0.50]), MAX_IMGSZ)
        changes: dict[str, Any] = {"imgsz": target} if imgsz < target else {}
        # close_mosaic 은 "마지막 N 에폭 동안 모자이크를 끈다" 는 뜻이라 epochs 보다 작아야 한다.
        # 크면 모자이크가 처음부터 끝까지 꺼져 작은 객체에 제일 도움되는 증강을 잃는다.
        close = min(10, max(0, epochs // 3))
        if close > int(params.get("close_mosaic", 0) or 0):
            changes["close_mosaic"] = close
        if imgsz < target:
            effect = (
                f"imgsz {imgsz} 에서는 특징이 거의 남지 않습니다. {target} 로 올리면 검출률이 "
                f"오르지만 시간과 VRAM 이 약 {round((target / imgsz) ** 2, 1)}배 늘어납니다."
            )
        else:
            # 해상도는 이미 충분하다. 그런데도 여기 온 것은 close_mosaic 때문이다.
            effect = "마지막 몇 에폭에서 모자이크를 끄면 실제 분포에 맞춰 마무리하는 효과가 있습니다."
        add(
            "tiny_objects",
            "warn",
            changes,
            f"박스의 {round(tiny * 100)}% 가 이미지 면적의 0.2% 미만입니다(작은 객체).",
            effect,
        )
    elif median_area > 0.20 and tiny < 0.05:
        add(
            "large_objects",
            "info",
            {"imgsz": 480},
            f"박스 중앙값이 이미지의 {round(median_area * 100)}% 로 큽니다.",
            "해상도를 낮춰도 검출률이 잘 떨어지지 않고 학습이 빨라집니다.",
        )

    # --- 데이터 양 ---
    if 0 < train_count < 500:
        changes = {}
        if epochs < 300:
            changes["epochs"] = 300
        if int(params.get("patience", 0) or 0) == 0:
            changes["patience"] = 50
        if float(params.get("mixup", 0.0) or 0.0) == 0.0:
            changes["mixup"] = 0.1
        add(
            "few_images",
            "warn",
            changes,
            f"학습 이미지가 {train_count}장으로 적습니다.",
            "에폭을 늘리고 증강을 강화하는 쪽이 유리합니다. 조기 종료를 함께 켜면 과적합 전에 멈춥니다.",
        )
    elif train_count > 20000:
        changes = {}
        if epochs > 80:
            changes["epochs"] = 80
        if params.get("cache") in (None, "False", False):
            changes["cache"] = "disk"
        add(
            "many_images",
            "info",
            changes,
            f"학습 이미지가 {train_count:,}장으로 많습니다.",
            "에폭당 시간이 길어 100에폭은 과합니다. 디스크 캐시를 켜면 로딩 병목이 줄어듭니다.",
        )

    # --- 판단만 돕고 값은 건드리지 않는 것들 ---
    ratio, most, least = _imbalance(report.get("class_instances") or {})
    if ratio > 20 and most and least:
        advisories.append(
            {
                "code": "class_imbalance",
                "severity": "warn",
                "message": f"'{most}' 가 '{least}' 보다 {round(ratio)}배 많습니다. "
                f"'{least}' 의 성능이 낮게 나올 수 있고, 그건 파라미터가 아니라 데이터로 풀어야 합니다.",
            }
        )

    total_images = int(report.get("total_images") or 0)
    missing = int((report.get("issue_counts") or {}).get("missing_label") or 0)
    if total_images and missing / total_images > 0.2:
        advisories.append(
            {
                "code": "many_background",
                "severity": "warn",
                "message": f"라벨 없는 이미지가 {round(missing / total_images * 100)}% 입니다. "
                f"의도한 배경 이미지가 아니라면 라벨이 누락된 것입니다.",
            }
        )

    # --- VRAM 은 추정기가 판단한다 ---
    merged = {
        **params,
        **{k: v["to"] for item in items for k, v in item["changes"].items()},
    }
    prediction = estimate.estimate(dataset, merged, devices)
    for warning in prediction.get("warnings", []):
        patch = warning.get("patch") or {}
        add(
            warning["code"],
            "warn" if warning["code"] == "vram_over" else "info",
            patch,
            warning["message"],
            "배치를 줄이면 VRAM 이 그만큼 줄고, 대신 에폭당 시간이 조금 늘어납니다.",
        )

    patch = {
        key: change["to"] for item in items for key, change in item["changes"].items()
    }
    # 마지막 관문. 여기서 걸리면 제안이 잘못된 것이므로 통째로 버린다 —
    # 사용자가 '적용' 을 누른 뒤 학습 시작이 422 로 죽는 것보다 낫다.
    try:
        param_schema.validate(patch, "params")
    except param_schema.ValidationError:
        return {"available": True, "items": [], "advisories": advisories, "patch": {}}

    return {"available": True, "items": items, "advisories": advisories, "patch": patch}
