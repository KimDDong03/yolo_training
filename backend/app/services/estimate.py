"""학습을 시작하기 전에 "얼마나 걸리고 VRAM 이 얼마나 드는지" 를 답한다.

첫 질문이 늘 이것인데, 지금은 첫 에폭이 끝나야 ETA 가 나온다. 그 전에는 알 수 없다.

방법은 해석적 추정 + 과거 실측 보정이다. 해석적 추정만으로는 절대값이 맞을 리 없지만
(GPU·데이터로더·디스크가 전부 다르다), **상대 관계**(imgsz 를 1.5배로 하면 시간이 2.25배)는
맞는다. 절대값은 이 PC 에서 실제로 끝난 학습의 epoch_time_s 로 보정한다.
그래서 학습을 한 번이라도 완주하면 추정이 눈에 띄게 정확해진다.

짧은 dry-run 으로 재는 방법도 있지만 torch import + 데이터셋 스캔에 30~60초가 걸리고
그동안 GPU 를 잡는다. 폼에 값을 입력할 때마다 그걸 돌릴 수는 없다.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from app.core import db
from app.services import gpu, run_manager

# 모델 스케일별 상대 연산량. yolo11n 을 1.0 으로 둔 근사값이며, 절대 시간은 보정이 맡는다.
MODEL_COST = {"n": 1.0, "s": 1.9, "m": 4.3, "l": 5.6, "x": 8.7}
# imgsz 640 · batch 1 기준 이미지당 활성값 메모리(GB) 근사.
VRAM_PER_IMAGE_GB = {"n": 0.09, "s": 0.16, "m": 0.32, "l": 0.44, "x": 0.66}
# 가중치·옵티마이저 상태 등 배치와 무관한 상주분(GB).
VRAM_BASE_GB = {"n": 0.7, "s": 1.1, "m": 1.9, "l": 2.6, "x": 3.6}

# 보정이 없을 때 쓰는 기준값: yolo11n · imgsz 640 · GPU 에서 이미지 한 장당 초.
BASE_SECONDS_PER_IMAGE = 0.004
# 이미지 수와 무관하게 에폭마다 드는 시간(초). 검증 pass · 플롯 · 체크포인트 저장 등.
#
# 이 항이 없으면 작은 데이터셋에서 보정이 무너진다. 16장짜리 run 의 실측 에폭이 1.3초인데
# 비례항만으로 예측하면 0.002초가 나오고, 보정 배수가 그 차이를 통째로 흡수해 500배가 된다.
# 그 배수를 imgsz 640 에 다시 적용하면 픽셀 항이 16배로 곱해져 22초/에폭 같은 값이 나온다.
FIXED_EPOCH_SECONDS = 1.5
# CPU 학습은 같은 조건에서 대략 이 배수만큼 느리다.
CPU_SLOWDOWN = 25.0
# 보정 배수가 이 범위를 벗어나면 표본이 지금 조건과 다른 영역에 있다고 보고 쓰지 않는다.
RATIO_BOUNDS = (0.2, 5.0)
# AMP 를 켜면 대략 이만큼으로 줄어든다 (시간·메모리 공통).
AMP_FACTOR = 0.6
# ultralytics AutoBatch 가 목표로 하는 VRAM 점유율.
AUTOBATCH_TARGET = 0.6
# 폼이 받는 배치 상한 (param_schema 의 batch max 와 같아야 한다).
MAX_BATCH = 1024

# 보정 표본을 모을 때 훑어볼 최근 완료 run 수. 전부 읽으면 폼 입력마다 디스크를 훑게 된다.
MAX_CALIBRATION_RUNS = 20
MIN_EPOCHS_FOR_SAMPLE = 3

_SCALE_RE = re.compile(r"yolo(?:v\d+|\d+)([nsmlx])\b", re.I)


def model_scale(model_ref: str) -> str | None:
    """파일 이름에서 모델 스케일(n/s/m/l/x)을 뽑는다. 모르면 None."""
    stem = Path(str(model_ref or "")).stem
    match = _SCALE_RE.search(stem)
    return match.group(1).lower() if match else None


def analytic_epoch_seconds(
    images: int, imgsz: int, scale: str, amp: bool, on_gpu: bool = True
) -> float:
    """보정 전 에폭 소요 시간. 절대값보다 조건 사이의 비율이 맞는 것이 목적이다.

    고정항 + 비례항으로 나눈다. 에폭마다 드는 검증·플롯 비용은 이미지 수에 비례하지 않는데,
    그걸 비례항에 밀어 넣으면 작은 데이터셋에서 보정 배수가 폭발한다.
    """
    pixels = (max(imgsz, 32) / 640.0) ** 2
    factor = AMP_FACTOR if amp else 1.0
    per_image = BASE_SECONDS_PER_IMAGE * MODEL_COST[scale] * pixels * factor
    if not on_gpu:
        per_image *= CPU_SLOWDOWN
    return FIXED_EPOCH_SECONDS + images * per_image


def analytic_vram_gb(batch: int, imgsz: int, scale: str, amp: bool) -> float:
    pixels = (max(imgsz, 32) / 640.0) ** 2
    factor = AMP_FACTOR if amp else 1.0
    return (
        VRAM_BASE_GB[scale] + VRAM_PER_IMAGE_GB[scale] * max(batch, 1) * pixels * factor
    )


def _epoch_samples(run_dir: Path) -> list[float]:
    """이 run 의 에폭별 실측 시간.

    첫 에폭은 버린다 — 웜업·AMP 체크·캐시 빌드가 섞여 유독 느리다(실측에서 5배 차이).
    final_val 은 학습 종료 후 검증이라 같은 에폭 번호로 한 번 더 찍히므로 세지 않는다.
    """
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return []
    seconds: list[float] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("t") != "epoch":
                    continue
                value = obj.get("epoch_time_s")
                if isinstance(value, (int, float)) and value > 0:
                    seconds.append(float(value))
    except OSError:
        return []
    return seconds[1:]


def _dataset_train_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in db.query("SELECT id, report FROM datasets"):
        try:
            report = json.loads(row["report"])
        except (TypeError, json.JSONDecodeError):
            continue
        count = report.get("train_count") or report.get("total_images")
        if isinstance(count, int) and count > 0:
            counts[row["id"]] = count
    return counts


def _calibration_samples() -> list[dict[str, Any]]:
    """과거 완료 run 에서 (실측 / 해석적 추정) 배수를 모은다.

    표본을 고르는 조건이 정확도를 좌우한다. 중간에 끊긴 학습, 여러 GPU, 데이터 일부만 쓴
    학습을 섞으면 배수가 망가진다.
    """
    train_counts = _dataset_train_counts()
    rows = db.query(
        "SELECT * FROM runs WHERE status = 'completed' ORDER BY created_at DESC LIMIT ?",
        (MAX_CALIBRATION_RUNS,),
    )

    samples: list[dict[str, Any]] = []
    for row in rows:
        run = db.row_to_run(row)
        params = run["params"]
        if float(params.get("fraction", 1.0) or 1.0) < 1.0:
            continue  # 데이터 일부만 쓴 학습은 다른 곡선이다
        if len(run["devices"]) > 1:
            continue  # DDP 는 확장 효율이 따로 논다
        images = train_counts.get(run["dataset_id"])
        scale = model_scale(params.get("model", ""))
        if not images or scale is None:
            continue

        measured = _epoch_samples(run_manager.run_dir_for(run["id"]))
        if len(measured) < MIN_EPOCHS_FOR_SAMPLE - 1:
            continue

        imgsz = int(params.get("imgsz", 640) or 640)
        amp = bool(params.get("amp", True))
        predicted = analytic_epoch_seconds(
            images, imgsz, scale, amp, bool(run["devices"])
        )
        if predicted <= 0:
            continue

        samples.append(
            {
                "scale": scale,
                "on_gpu": bool(run["devices"]),
                "ratio": statistics.median(measured) / predicted,
            }
        )
    return samples


def _calibration(scale: str, on_gpu: bool) -> tuple[float, int, str]:
    """(배수, 표본수, 출처). 쓸 만한 표본이 없으면 (1.0, 0, "analytic")."""
    samples = _calibration_samples()
    for pool in (
        [s for s in samples if s["scale"] == scale and s["on_gpu"] == on_gpu],
        [s for s in samples if s["on_gpu"] == on_gpu],
    ):
        if not pool:
            continue
        ratio = statistics.median(s["ratio"] for s in pool)
        # 배수가 이 범위를 벗어나면 모델이 표본의 조건을 설명하지 못한다는 뜻이다.
        # 그런 배수를 다른 조건에 곱하면 엉뚱한 값이 나오므로 차라리 보정을 포기한다.
        if RATIO_BOUNDS[0] <= ratio <= RATIO_BOUNDS[1]:
            return ratio, len(pool), "calibrated"
    return 1.0, 0, "analytic"


def _total_vram_gb(devices: list[int]) -> float | None:
    if not devices:
        return None
    by_index = {int(g["index"]): g for g in gpu.list_gpus()}  # type: ignore[arg-type]
    chosen = by_index.get(devices[0])
    if not chosen:
        return None
    return round(int(chosen["memory_total_mb"]) / 1024, 2)  # type: ignore[arg-type]


def estimate(
    dataset: dict[str, Any], params: dict[str, Any], devices: list[int]
) -> dict[str, Any]:
    report = dataset.get("report") or {}
    images = report.get("train_count") or report.get("total_images") or 0
    scale = model_scale(params.get("model", ""))

    if not images or scale is None:
        # 모델 이름에서 스케일을 못 읽으면(사용자 지정 .yaml 등) 추정 근거가 없다.
        # 틀린 숫자를 보여주느니 없다고 말한다.
        return {
            "ok": False,
            "reason": "학습 이미지 수 또는 모델 크기를 알 수 없어 예측할 수 없습니다.",
        }

    imgsz = int(params.get("imgsz", 640) or 640)
    epochs = int(params.get("epochs", 100) or 100)
    amp = bool(params.get("amp", True))
    on_gpu = bool(devices)
    total_vram = _total_vram_gb(devices)

    assumptions: list[str] = []
    batch = int(params.get("batch", -1) or -1)
    if batch <= 0:
        # AutoBatch 가 실행 시점에 정하므로 지금은 알 수 없다. VRAM 목표치에서 역산하고
        # 그 가정을 반드시 밝힌다 — 안 밝히면 빗나갔을 때 기능 전체의 신뢰가 사라진다.
        if total_vram:
            per_image = (
                VRAM_PER_IMAGE_GB[scale]
                * ((imgsz / 640.0) ** 2)
                * (AMP_FACTOR if amp else 1.0)
            )
            usable = max(total_vram * AUTOBATCH_TARGET - VRAM_BASE_GB[scale], 0.1)
            batch = max(1, int(usable / max(per_image, 1e-6)))
        else:
            batch = 16
        # VRAM 만 보면 작은 이미지에서 수천이 나온다. 한 배치가 데이터셋보다 클 수는 없고,
        # 폼이 받는 상한(param_schema 의 batch max)도 넘을 수 없다.
        batch = max(1, min(batch, images, MAX_BATCH))
        assumptions.append(
            f"배치가 자동(-1)이라 {batch} 로 가정했습니다. 실제 값은 학습 시작 시 정해집니다."
        )

    ratio, samples, source = _calibration(scale, on_gpu)
    epoch_seconds = analytic_epoch_seconds(images, imgsz, scale, amp, on_gpu) * ratio
    total_seconds = epoch_seconds * max(epochs, 1)
    vram = round(analytic_vram_gb(batch, imgsz, scale, amp), 2) if on_gpu else None

    device_label = f"GPU {devices[0]}" if on_gpu else "CPU"
    assumptions.insert(
        0,
        f"yolo11{scale} · 이미지 {images}장 · imgsz {imgsz} · 배치 {batch} · {device_label}",
    )
    if source == "calibrated":
        assumptions.append(
            f"이 PC 에서 완료된 학습 {samples}개의 실측 시간으로 보정했습니다."
        )
        spread = (0.7, 1.4)
    else:
        assumptions.append(
            "아직 완료된 학습이 없어 보정하지 못했습니다. 오차가 클 수 있습니다."
        )
        spread = (0.4, 2.5)
    assumptions.append("검증(val) 시간과 첫 에폭 웜업이 포함된 값입니다.")

    warnings: list[dict[str, Any]] = []
    level = "ok"
    if vram is not None and total_vram:
        used = vram / total_vram
        if used > 1.0:
            level = "over"
            warnings.append(
                {
                    "code": "vram_over",
                    "message": f"예상 VRAM {vram}GB 가 이 GPU 의 {total_vram}GB 를 넘습니다. "
                    f"그대로 시작하면 메모리 부족으로 실패할 가능성이 높습니다.",
                    "patch": {"batch": max(1, batch // 2)},
                }
            )
        elif used > 0.9:
            level = "tight"
            warnings.append(
                {
                    "code": "vram_tight",
                    "message": f"예상 VRAM {vram}GB 가 가용량의 {round(used * 100)}% 입니다. 여유가 거의 없습니다.",
                    "patch": {"batch": max(1, batch // 2)},
                }
            )

    return {
        "ok": True,
        "epoch_time_s": round(epoch_seconds, 1),
        "total_time_s": round(total_seconds, 1),
        "range_s": [round(total_seconds * spread[0]), round(total_seconds * spread[1])],
        "batch_effective": batch,
        "vram_gb": vram,
        "vram_total_gb": total_vram,
        "vram_level": level,
        "source": source,
        "samples": samples,
        "assumptions": assumptions,
        "warnings": warnings,
    }
