"""하이퍼파라미터 자동 탐색(Phase 4) — 탐색 공간, 리포트 조립, 소요 추정.

ultralytics 의 `model.tune()` 을 그대로 쓴다(반입 0). 이 모듈은 그 주변만 맡는다:

1. **탐색 공간을 좁힌다.** 기본 공간은 26차원인데 반복 10~20회로는 랜덤 변이가 잡음이다.
   더 중요한 이유는 따로 있다 — 기본 공간의 `cls_pw` / `bgr` / `cutmix` 는 param_schema 에
   없어서 탐색해 봐야 새 학습 폼에 넣을 수가 없다. 여기 있는 9개는 전부 폼이 받는 값이다.
2. **NDJSON 을 리포트로 옮긴다.** Tuner 는 반복 단위 콜백이 없어서(원본 확인) 진행 상황을
   알 방법이 `tune_results.ndjson` 폴링뿐이다.
3. **소요 시간을 추정한다.** estimate.estimate() 는 1회 학습·전체 데이터 기준이라
   그대로 쓰면 틀린다. 여기서 반복 수와 데이터 비율을 곱하고 가정을 밝힌다.

리포트에 **잡 상태(running/stopped/failed)를 담지 않는다.** 그건 JobStatus 가 단일 원천이고,
복제하면 강제 종료 시 리포트만 영원히 "진행 중" 으로 남아 서로를 부정한다.
부분 완료는 iterations_done < iterations_target 이 말한다.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from app.services import estimate, param_schema

# (min, max, gain?) — ultralytics Tuner 의 space 형식 그대로다.
# 전부 param_schema._SPEC 에 있고 범위도 스키마 min/max 안이다(test_tune.py 가 고정한다).
SPACE: dict[str, tuple[float, ...]] = {
    "lr0": (1e-5, 1e-2),
    "lrf": (0.01, 1.0),
    "momentum": (0.7, 0.98, 0.3),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "box": (1.0, 20.0),
    "cls": (0.1, 4.0),
    "dfl": (0.4, 12.0),
    "mosaic": (0.0, 1.0),
}

# 처방의 **바닥선**이다. 이만큼도 못 올렸으면 무엇을 재든 처방하지 않는다.
#
# 단위는 **mAP50-95** 다. ultralytics 8.4.47 의 검출 fitness 가중치는 [P,R,mAP50,mAP50-95] =
# [0,0,0,1] 이라(utils/metrics.py:965) fitness 가 곧 mAP50-95 다. 예전 판의 0.1/0.9 가중합이
# 아니므로 "fitness 0.02" 는 "mAP50-95 2%p" 로 그대로 읽으면 된다.
#
# **이 상수만으로 판정하지 않는다.** 실측(.codex/phase-4.md "시드 변동폭")에서 같은
# 하이퍼파라미터도 시드만 바꾸면 크게 흔들렸다 — 에폭 3·데이터 15% 에서 표준편차 0.026,
# 에폭 10·데이터 100% 에서 0.007. 3.5배 차이라 어떤 상수를 골라도 한쪽에서는 틀린다.
# 그래서 실제 문턱은 **탐색이 자기 설정에서 직접 잰 노이즈**와 이 바닥선 중 큰 쪽이다
# (아래 actionable_threshold).
MIN_ACTIONABLE_GAIN = 0.005

# 확인 시도에 쓸 시드들. 시도는 전부 0 으로 도므로(job.log 로 확인) 겹치지 않으면 된다.
#
# **왜 하나가 아니라 셋인가.** 처음에는 하나만 다시 돌려 그 차이를 문턱으로 썼다. 실측이
# 그것을 반증했다 — 기준(시드 0) 0.13099 대 확인(시드 1) 0.13922 로 차이가 0.0082 밖에
# 안 나왔는데, 같은 설정 6시드의 진짜 표준편차는 0.0258 이었다. 시드 1 이 우연히 가깝게
# 나왔을 뿐이고, 그 낮은 문턱이 정확히 걸러야 할 상승(+0.0189)을 통과시켰다.
# 두 표본의 차 |Δ| 는 평균이 1.13σ 지만 표본이 하나면 그 자체가 크게 흔들린다.
# 기준까지 합쳐 네 표본이면 표준편차를 직접 추정할 수 있다.
PROBE_SEEDS = (1, 2, 3)


def selection_factor(trials: int) -> float:
    """N번 중 최고를 골랐을 때, 효과가 없어도 기대되는 상승폭(표준편차의 배수).

    탐색이 보고하는 것은 **N개 중 최고**다. 하이퍼파라미터가 아무 일도 하지 않아도 N번 뽑으면
    그중 제일 높은 값은 평균보다 위에 있다 — 많이 뽑을수록 더. 정규분포에서 그 기대치는
    대략 표준편차 × sqrt(2 ln N) 이다(N=4 면 1.67배, N=20 이면 2.45배).

    이 보정이 없으면 문턱이 사실상 무력하다. 실측으로 확인했다 — 시도 4회에서 표준편차가
    0.0140 인데 최고 상승폭이 0.0189 였다. 표준편차를 그대로 문턱으로 쓰면 통과하지만,
    "4번 뽑아 제일 좋은 것" 이라는 사실을 넣으면 0.0233 이 되어 걸린다.
    """
    return math.sqrt(2.0 * math.log(max(trials, 2)))


def actionable_threshold(spread: float | None, trials: int = 2) -> float:
    """이번 탐색에서 처방을 낼 최소 상승폭.

    노이즈를 쟀으면 (표준편차 × 선택 보정)과 바닥선 중 큰 쪽을 쓴다. 같은 하이퍼파라미터를
    시드만 바꿔 돌렸을 때 벌어지는 만큼은 하이퍼파라미터가 한 일이 아니고, 여러 번 뽑아
    제일 좋은 것을 고른 몫도 하이퍼파라미터가 한 일이 아니다.
    """
    if spread is None:
        return MIN_ACTIONABLE_GAIN
    return max(MIN_ACTIONABLE_GAIN, float(spread) * selection_factor(trials))

# 시도마다 붙는 고정 비용(초). 에폭 수와 무관하게 든다.
#
# **실측값이다.** african-wildlife 3시도 × 5에폭 × fraction 0.2 (RTX 3060) 에서
# 전체 298초, 학습 내부 경과 합 121초 → 시도당 기동 비용 약 57초(프로세스 띄우기, torch/ultralytics
# import, 데이터 스캔, 체크포인트 로드, plot_tune_results). 여기에 첫 에폭 웜업이 얹힌다
# (첫 에폭 24.7초 대 정상 3.3초 → 약 21초). 짧은 시도에서는 이 둘이 학습 시간보다 크다.
TRIAL_OVERHEAD_S = 80.0

SCHEMA_VERSION = 1


def schema_violations() -> list[str]:
    """SPACE 가 param_schema 밖으로 나가는지 검사한다. 나가면 patch 가 422 로 죽는다."""
    index = param_schema.field_index()
    problems: list[str] = []
    for key, bounds in SPACE.items():
        field = index.get(key)
        if field is None or field.get("scope") != "params":
            problems.append(f"{key}: param_schema 의 학습 파라미터가 아닙니다.")
            continue
        low, high = float(bounds[0]), float(bounds[1])
        f_min, f_max = field.get("min"), field.get("max")
        if f_min is not None and low < float(f_min):
            problems.append(f"{key}: 탐색 하한 {low} 이 스키마 하한 {f_min} 보다 작습니다.")
        if f_max is not None and high > float(f_max):
            problems.append(f"{key}: 탐색 상한 {high} 이 스키마 상한 {f_max} 보다 큽니다.")
    return problems


# ------------------------------------------------------------------ NDJSON 읽기

def results_path(tune_dir: Path) -> Path:
    return Path(tune_dir) / "tune_results.ndjson"


def read_results(tune_dir: Path) -> list[dict[str, Any]]:
    """완료된 시도 기록을 읽는다.

    **깨진 줄은 건너뛴다.** 폴러는 ultralytics 가 append 하는 도중에도 이 파일을 읽으므로
    마지막 줄이 잘려 있을 수 있다. 그 한 줄 때문에 리포트 전체를 잃으면 안 된다.
    """
    path = results_path(tune_dir)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("iteration") is not None:
            records.append(record)
    records.sort(key=lambda r: int(r.get("iteration") or 0))
    return records


def repair_results(tune_dir: Path) -> int:
    """이어하기 전에 NDJSON 의 깨진 꼬리를 잘라 낸다. 버린 줄 수를 돌려준다.

    **우리가 관대한 것만으로는 부족하다.** `read_results` 는 깨진 줄을 건너뛰지만, 이어하기를
    실제로 수행하는 것은 ultralytics 이고 그쪽 `Tuner._load_local_results` 는
    `json.loads(line)` 을 그대로 돌린다 — 오류 처리가 없다. 강제 종료가 append 도중에
    떨어지면 마지막 줄이 잘리고, 그 파일로는 **이어하기가 영원히 JSONDecodeError 로 죽는다.**
    강제 종료 + 이어하기가 이 잡의 정상 사용법이므로 여기서 고쳐 놓고 넘긴다.

    깨진 줄부터 뒤는 통째로 버린다. 중간 한 줄만 빼면 남은 기록의 iteration 번호와
    ultralytics 가 세는 개수가 어긋난다.
    """
    path = results_path(tune_dir)
    if not path.is_file():
        return 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 0

    keep: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            break
        keep.append(line)

    dropped = len([line for line in lines if line.strip()]) - len(keep)
    if dropped > 0:
        # 제자리에 덮어쓰지 않는다. 쓰는 도중에 죽으면 **멀쩡한 기록까지** 잘린 파일이 되어,
        # 깨진 꼬리 하나를 고치려다 몇 시간치를 날린다. 임시 파일에 다 쓰고 통째로 바꿔 끼운다.
        tmp = path.with_name(path.name + ".repair")
        tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        os.replace(tmp, path)
    return dropped


def _finite(value: Any) -> float | None:
    """숫자이고 유한하면 float, 아니면 None.

    NaN 을 그대로 통과시키면 두 가지가 깨진다. (1) `json.dumps` 가 표준이 아닌 `NaN` 리터럴을
    쓰고 브라우저의 `JSON.parse` 가 죽는다 — 이 레포가 Phase 0 에서 이미 한 번 당한 버그다.
    (2) NaN 과의 비교는 전부 False 라 개선폭 판정이 조용히 어긋난다.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metrics_of(record: dict[str, Any]) -> dict[str, float]:
    """ultralytics 는 데이터셋 이름을 키로 한 겹 감싸 둔다. 데이터셋이 하나면 벗겨 낸다."""
    datasets = record.get("datasets")
    if isinstance(datasets, dict) and len(datasets) == 1:
        inner = next(iter(datasets.values()))
        if isinstance(inner, dict):
            cleaned = {key: _finite(value) for key, value in inner.items()}
            return {key: value for key, value in cleaned.items() if value is not None}
    return {}


def _trial(record: dict[str, Any]) -> dict[str, Any]:
    """한 시도. **실패한 시도를 성공으로 읽지 않는 것이 여기의 요점이다.**

    ultralytics 는 학습이 예외로 죽어도 그 시도를 `{"fitness": 0.0}` 으로 기록하고 다음으로
    넘어간다(tuner.py:511-512). 그걸 정상값으로 받으면 **시도 1(기준)이 실패했을 때 0.0 대비
    개선폭이 잡혀 근거 없는 처방이 나간다.** 성공한 시도는 precision/recall 같은 지표를
    함께 남기므로, fitness 말고 다른 지표가 있는지로 가른다.
    """
    raw = record.get("hyperparameters")
    raw = raw if isinstance(raw, dict) else {}
    # 하이퍼파라미터도 유한성을 본다. NaN 이 하나 섞이면 allow_nan=False 인 리포트 쓰기가
    # 통째로 실패해 진행 표시가 멈춘다 — 그 한 시도만 실패로 두는 편이 낫다.
    hyp = {key: _finite(value) for key, value in raw.items() if key in SPACE}
    metrics = _metrics_of(record)
    fitness = _finite(record.get("fitness"))
    return {
        "i": int(record.get("iteration") or 0),
        "fitness": fitness if fitness is not None else 0.0,
        "hyp": {key: value for key, value in hyp.items() if value is not None},
        "metrics": metrics,
        "ok": (
            fitness is not None
            and bool(set(metrics) - {"fitness"})
            and all(value is not None for value in hyp.values())
        ),
    }


# ------------------------------------------------------------------ 리포트

def build_report(
    tune_dir: Path,
    args: dict[str, Any],
    elapsed_s: float | None = None,
    resumed: int = 0,
    noise: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """NDJSON 을 화면이 그대로 쓸 수 있는 리포트로 옮긴다. 실행 중에도 호출된다.

    elapsed_s / resumed 를 주면 **실측 경과로** 남은 시간을 낸다. 시작 전 추정(estimate_total)
    과 달리 모델이 아니라 이 잡이 실제로 쓴 시간이라, 몇 시간짜리 잡에서는 이쪽이 진실이다.
    resumed 는 이번 실행이 물려받은 시도 수다 — 그만큼은 이번 경과에 포함되지 않았다.
    """
    target = int(args.get("iterations") or 0)
    trials = [_trial(record) for record in read_results(tune_dir)]

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "iterations_done": len(trials),
        "iterations_target": target,
        "space": {k: list(v) for k, v in SPACE.items()},
        "args": dict(args),
        "min_gain": MIN_ACTIONABLE_GAIN,
        "trials": trials,
        "baseline": None,
        "best": None,
        "gain": None,
        "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
        "trial_time_s": None,
        "eta_s": None,
        # 이번 설정에서 직접 잰 시드 노이즈와, 그로부터 정해진 실제 문턱.
        # 문턱은 시도 수에 따라 달라지므로(선택 보정) 아래에서 다시 정한다.
        "noise": noise,
        "threshold": MIN_ACTIONABLE_GAIN,
        "available": False,
        "items": [],
        "advisories": [],
        "patch": {},
    }

    # 이번 실행에서 실제로 끝낸 시도로 나눈다. 이어받은 시도는 이번 경과에 없다.
    fresh = len(trials) - max(resumed, 0)
    if elapsed_s is not None and fresh > 0:
        per_trial = elapsed_s / fresh
        report["trial_time_s"] = round(per_trial, 1)
        report["eta_s"] = round(per_trial * max(target - len(trials), 0), 1)

    if not trials:
        report["reason"] = "아직 끝난 시도가 없습니다."
        return report

    # 실패한 시도는 비교에서 뺀다. 목록에는 남겨 사용자가 몇 개가 깨졌는지 보게 한다.
    usable = [t for t in trials if t["ok"]]
    failed = len(trials) - len(usable)
    if failed:
        report["advisories"].append(
            f"시도 {failed}개가 학습 중 실패해 비교에서 제외했습니다."
        )

    # 시도 1 은 변이 없는 기본값이다(Tuner._mutate 는 기록이 비면 기본값으로 떨어진다).
    # 그래서 "기본값 대비 얼마나 올랐나" 를 지어내지 않고 잴 수 있다.
    baseline = next((t for t in usable if t["i"] == 1), None)
    best = max(usable, key=lambda t: t["fitness"]) if usable else None
    report["baseline"] = baseline
    report["best"] = best

    if baseline is None or best is None:
        # 기준 시도가 없거나 실패했다. 실패한 기준(fitness 0.0)에 대고 개선폭을 재면
        # 아무 시도나 대단해 보인다 — 근거 없는 처방이 나가느니 아무 말도 하지 않는다.
        report["reason"] = (
            "기준이 되는 첫 시도가 없거나 실패해 기본값 대비 개선폭을 계산할 수 없습니다."
        )
        return report

    gain = round(best["fitness"] - baseline["fitness"], 5)
    report["gain"] = gain
    report["available"] = True

    if len(usable) < 2:
        report["advisories"].append(
            "아직 기준 시도 하나뿐입니다. 비교할 대상이 생기면 제안이 나옵니다."
        )
        return report

    measured = (noise or {}).get("stdev")
    threshold = actionable_threshold(measured, len(usable))
    report["threshold"] = round(threshold, 5)
    if measured is not None:
        count = len((noise or {}).get("fitness") or []) + 1
        report["advisories"].append(
            f"같은 값을 시드만 바꿔 {count}번 돌려 봤더니 결과가 ±{float(measured):.4f} 만큼"
            f" 흔들렸습니다. 시도 {len(usable)}개 중 최고를 고른 몫까지 감안해"
            f" +{threshold:.4f} 넘게 올라야 하이퍼파라미터 덕이라고 봅니다."
        )

    if gain < threshold:
        if measured is not None and gain >= MIN_ACTIONABLE_GAIN:
            # 바닥선은 넘었지만 이번 설정의 노이즈 안이다. 왜 안 되는지를 정확히 말한다 —
            # 이 문장이 없으면 사용자는 "+0.02 나 올랐는데 왜 제안이 없나" 로 읽는다.
            report["advisories"].append(
                f"최고 조합이 기본값보다 +{gain:.4f} 높지만 위 흔들림(±{float(measured):.4f})보다 "
                "작아 우연과 구분되지 않습니다. 시도당 에폭이나 데이터 비율을 늘리면 "
                "이 흔들림이 줄어 판단할 수 있게 됩니다."
            )
        else:
            report["advisories"].append(
                f"{len(trials)}번 시도했지만 기본값보다 의미 있게 나은 조합을 찾지 못했습니다"
                f"(최고 +{gain:.4f}, 기준 {threshold:.4f}). 지금 값을 그대로 쓰십시오."
            )
        return report

    patch = {k: v for k, v in best["hyp"].items() if k in SPACE}
    try:
        # 마지막 관문. 여기서 걸리면 제안이 잘못된 것이므로 통째로 버린다 —
        # 사용자가 '적용' 을 누른 뒤 학습 시작이 422 로 죽는 것보다 낫다. (recommend.py 와 같은 규칙)
        patch = param_schema.validate(patch, "params")
    except param_schema.ValidationError as exc:
        report["advisories"].append(f"찾은 값을 폼에 넣을 수 없어 제안하지 않습니다: {exc}")
        return report

    report["patch"] = patch
    report["items"] = [
        {
            "rule": "tuned",
            "severity": "info",
            "changes": {
                key: {"from": baseline["hyp"].get(key), "to": value}
                for key, value in patch.items()
            },
            "reason": (
                f"{len(trials)}번 시도 중 {best['i']}번째가 가장 좋았습니다"
                f"(mAP50-95 {best['fitness']:.4f}, 기본값 {baseline['fitness']:.4f}, +{gain:.4f})."
                + (
                    f" 같은 설정의 시드 흔들림 ±{float(measured):.4f} 보다 큽니다."
                    if measured is not None
                    else ""
                )
            ),
            "effect": (
                "이 값으로 본 학습을 돌리면 같은 방향의 개선을 기대할 수 있습니다. "
                "탐색은 짧은 학습으로 순위를 매긴 것이라 개선폭 자체는 달라질 수 있습니다."
            ),
        }
    ]
    return report


# ------------------------------------------------------------------ 설정 경고


def setting_warnings(fraction: float, epochs: int) -> list[str]:
    """시도를 짧게 잡았을 때 무엇을 잃는지. 실측 근거를 그대로 문장에 넣는다.

    측정: african-wildlife · yolo11n · imgsz 640 · batch 16 · RTX 3060, 시드 4~6개씩.

        에폭 3 · 데이터 15%   표준편차 0.026 · 시도 약 90초 · 평균 mAP50-95 0.137
        에폭 3 · 데이터 100%  표준편차 0.012 · 시도 약 110초 · 평균 0.683
        에폭 10 · 데이터 100% 표준편차 0.007 · 시도 약 192초 · 평균 0.753

    데이터를 줄이는 쪽이 노이즈에 더 크게 기여하는데(2.1배 대 1.6배) 아끼는 시간은 가장 적다.
    시도당 고정 비용(기동 + 첫 에폭 웜업)이 지배해서다.
    """
    warnings: list[str] = []
    if fraction < 1.0:
        warnings.append(
            f"데이터를 {int(fraction * 100)}% 만 쓰면 시간은 조금밖에 줄지 않는데"
            " 결과의 흔들림은 크게 늘어납니다. 실측에서 15% 로 줄였을 때 시도 시간은"
            " 90초 대 110초로 18% 아꼈지만 시드에 따른 흔들림은 2배가 됐습니다."
            " 게다가 그 상태의 모델은 거의 학습되지 않아(mAP50-95 0.14 대 0.68) 거기서 매긴"
            " 순위가 본 학습으로 이어진다는 보장이 없습니다. 시간을 줄이려면 시도 횟수를"
            " 줄이는 편이 낫습니다."
        )
    if epochs < 5:
        warnings.append(
            f"시도당 에폭이 {epochs} 면 짧아 조합 간 차이가 우연에 묻히기 쉽습니다."
            " 실측에서 에폭 3 대 10 의 흔들림이 0.012 대 0.007 이었습니다."
        )
    return warnings


# ------------------------------------------------------------------ 소요 추정

def estimate_total(
    dataset: dict[str, Any], args: dict[str, Any], devices: list[int]
) -> dict[str, Any]:
    """탐색 전체의 예상 소요. estimate.estimate() 를 시도 1회분으로 쓰고 반복 수를 곱한다."""
    iterations = max(int(args.get("iterations") or 1), 1)
    epochs = max(int(args.get("epochs") or 1), 1)
    fraction = float(args.get("fraction") or 1.0)

    params = {
        "model": args.get("model", ""),
        "imgsz": args.get("imgsz", 640),
        "batch": args.get("batch", -1),
        "epochs": epochs,
        "amp": True,
    }
    single = estimate.estimate(dataset, params, devices)
    if not single.get("ok"):
        return {"ok": False, "reason": single.get("reason", "예상 소요를 계산할 수 없습니다.")}

    epoch_seconds = float(single["epoch_time_s"]) * max(min(fraction, 1.0), 0.01)
    trial_seconds = epoch_seconds * epochs + TRIAL_OVERHEAD_S
    total_seconds = trial_seconds * iterations

    assumptions = [
        f"시도 {iterations}회 × 에폭 {epochs} · 데이터 {int(fraction * 100)}%",
        "데이터 비율은 시간에 비례한다고 가정했습니다.",
        f"시도마다 준비 비용 약 {int(TRIAL_OVERHEAD_S)}초를 더했습니다.",
    ]
    if len(devices) > 1:
        # 다중 GPU 가속을 추정에 넣지 않는다. 이 머신에서 측정한 적이 없어서 배수를 지어내야 한다.
        # 느린 쪽으로 틀리는 편이 낫다.
        assumptions.append(
            f"GPU {len(devices)}장을 쓰지만 가속은 반영하지 않았습니다 — 실제로는 더 빠를 수 있습니다."
        )
    assumptions.extend(single.get("assumptions", []))

    return {
        "ok": True,
        "warnings": setting_warnings(fraction, epochs),
        "trial_time_s": round(trial_seconds, 1),
        "total_time_s": round(total_seconds, 1),
        "epoch_time_s": round(epoch_seconds, 2),
        "iterations": iterations,
        "source": single.get("source"),
        "assumptions": assumptions,
        "note": "근사치입니다. 실제 소요는 데이터와 하드웨어에 따라 달라집니다.",
    }
