"""Phase 6 캠페인의 결과를 걷어 블록별 판정표를 낸다. 읽기 전용이다.

    python scripts/campaign_report.py          # 전체
    python scripts/campaign_report.py G B      # 블록만

**상태를 따로 쌓지 않는다.** 진실은 이미 두 곳에 있다 - DB 의 runs 테이블(이름·상태)과
storage/runs/<id>/events.jsonl(지표·시간·VRAM). 별도 원장을 만들면 그 원장과 실제 run 이
어긋날 수 있고, 그 자체가 새 실패 모드다. 행렬 좌표는 run 이름에 인코딩돼 있다.

수확은 반드시 end 이벤트의 summary["mAP50-95"] 로 한다. final_val 의 fitness 필드는
훅이 읽는 trainer.fitness 가 한 스텝 뒤처져 1e-4 쯤 어긋난다(실측: 0.80019 대 0.8000737).
조용히 틀리는 종류라 쓰지 않는다.

콘솔이 cp949 라 출력은 ASCII 로 한다.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BASE_DIR / "storage" / "runs"
DB_PATH = BASE_DIR / "app.db"

# 실용적 하한. tune.MIN_ACTIONABLE_GAIN 과 같은 값이며 이유도 같다 -
# 잰 흔들림이 0 이어도 이보다 작은 상승은 처방하지 않는다.
MIN_ACTIONABLE_GAIN = 0.005

# 단측 95% t 값. scipy 를 들이지 않으려고 필요한 자유도만 적어 둔다.
T_ONE_SIDED_95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015}


def threshold(base: list[float], arm: list[float]) -> float:
    """이 비교에서 넘어야 하는 상승폭.

    **왜 표준편차를 그대로 쓰지 않는가.** 비교하는 차이에는 독립적인 흔들림이 둘 들어 있다 -
    arm 쪽과 기준 쪽. 여기에 표본이 몇 개뿐이라 표준편차 추정 자체가 흔들리는 몫을
    t 분포로 얹는다.

    arm 도 여러 시드로 돌았으면 **두 표본을 다 쓴다**(Welch). arm 을 1회 관측으로 치면
    그쪽 흔들림을 기준 쪽 sigma 로 대신 잡게 되는데, 실측에서 그 둘이 크게 다르다 -
    yolo11m 세 시드의 폭이 0.0017 인데 yolo11n 은 0.0258 이었다. 같은 데이터로 문턱이
    두 배 좁아지므로 그냥 버릴 이유가 없다.

    tune.actionable_threshold 를 그대로 쓰지 않는 이유: 그 함수의 보정은 "N개 중 최고를
    골랐을 때" 를 위한 것이고(tune.py:71-82), 여기 비교는 사전에 정한 2개 arm 이라
    최댓값 선택이 없다. 같은 함수를 계약 밖에서 쓰면 숫자에 해석이 없어진다.
    """
    k = len(base)
    var_base = statistics.variance(base) / k

    if len(arm) >= 2:
        m = len(arm)
        var_arm = statistics.variance(arm) / m
        se = math.sqrt(var_base + var_arm)
        # Welch-Satterthwaite. 분모가 0 이면(두 표본 모두 변동 0) 기준 쪽 자유도로 떨어진다.
        denominator = var_base**2 / (k - 1) + var_arm**2 / (m - 1)
        df = (var_base + var_arm) ** 2 / denominator if denominator > 0 else k - 1
    else:
        # arm 이 한 번뿐이면 그쪽 흔들림을 기준 쪽 sigma 로 대신 잡는 수밖에 없다.
        se = math.sqrt(var_base * (k + 1))
        df = k - 1

    t = T_ONE_SIDED_95.get(int(df), 1.96)
    return max(MIN_ACTIONABLE_GAIN, t * se)


def run_names() -> dict[str, str]:
    """{run id: 이름}. 이름이 곧 (블록, arm, 시드) 좌표다."""
    import sqlite3

    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        return {
            row[0]: row[1]
            for row in conn.execute("SELECT id, name FROM runs")
            if str(row[1]).startswith("p6/")
        }


def harvest(run_id: str) -> dict | None:
    """이 run 의 events.jsonl 에서 필요한 것만 뽑는다."""
    path = RUNS_DIR / run_id / "events.jsonl"
    if not path.is_file():
        return None

    epochs, end, start = [], None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("t")
        if kind == "epoch":
            epochs.append(event)
        elif kind == "end":
            end = event
        elif kind == "start":
            start = event

    if end is None or end.get("status") != "completed":
        return None

    # 1에폭은 버린다. 웜업이 섞여 있어 정상 상태가 아니다 (estimate._epoch_samples 와 같은 규칙).
    times = [e["epoch_time_s"] for e in epochs[1:] if isinstance(e.get("epoch_time_s"), (int, float))]
    memory = [e["mem_gb"] for e in epochs if isinstance(e.get("mem_gb"), (int, float))]

    # best 가 몇 번째 에폭에서 나왔는가. arm 마다 에폭 수가 다를 때 그 차이가 진짜 학습
    # 이득인지 "더 여러 번 뽑았을 뿐" 인지를 가르는 진단이다.
    best_epoch, best_seen = None, None
    for event in epochs:
        fitness = event.get("best_fitness")
        if isinstance(fitness, (int, float)) and (best_seen is None or fitness > best_seen):
            best_seen, best_epoch = fitness, event.get("epoch")

    summary = end.get("summary") or {}
    return {
        "map": summary.get("mAP50-95"),
        "map50": summary.get("mAP50"),
        "epochs_done": end.get("epochs_done"),
        "epoch_s": statistics.median(times) if times else None,
        "mem_gb": max(memory) if memory else None,
        "batch": (start or {}).get("batch"),
        "imgsz": (start or {}).get("imgsz"),
        "best_epoch": best_epoch,
    }


def collect() -> dict[str, dict[str, list[dict]]]:
    """{블록: {arm: [결과, ...]}}"""
    blocks: dict[str, dict[str, list[dict]]] = {}
    for run_id, name in run_names().items():
        parts = name.split("/")
        if len(parts) != 4:
            continue
        _, block, arm, seed = parts
        result = harvest(run_id)
        if result is None:
            continue
        result["seed"] = seed
        blocks.setdefault(block, {}).setdefault(arm, []).append(result)
    for arms in blocks.values():
        for rows in arms.values():
            rows.sort(key=lambda r: r["seed"])
    return blocks


# 블록마다 무엇이 기준 arm 인가. 없으면 mAP 판정을 하지 않고 표만 낸다.
BASELINE = {"F": "balanced", "G": "yolo11n", "B": "off640", "D": "off", "H": "off"}


def show_measurements(block: str, arms: dict[str, list[dict]]) -> None:
    print(f"=== block {block}")
    print(f"  {'arm':12s} {'seed':>4s} {'mAP50-95':>9s} {'mAP50':>7s} "
          f"{'ep_s':>7s} {'mem_gb':>7s} {'batch':>5s} {'imgsz':>5s} {'ep_done':>7s} {'best_ep':>7s}")
    for arm, rows in sorted(arms.items()):
        for row in rows:
            def fmt(key: str, spec: str) -> str:
                value = row.get(key)
                return format(value, spec) if isinstance(value, (int, float)) else "-"
            print(f"  {arm:12s} {row['seed']:>4s} {fmt('map', '9.5f')} {fmt('map50', '7.4f')} "
                  f"{fmt('epoch_s', '7.2f')} {fmt('mem_gb', '7.3f')} {fmt('batch', '5.0f')} "
                  f"{fmt('imgsz', '5.0f')} {fmt('epochs_done', '7.0f')} {fmt('best_epoch', '7.0f')}")


def show_verdict(block: str, arms: dict[str, list[dict]]) -> None:
    baseline_arm = BASELINE.get(block)
    if baseline_arm is None or baseline_arm not in arms:
        return

    scores = [r["map"] for r in arms[baseline_arm] if isinstance(r.get("map"), (int, float))]
    if len(scores) < 3:
        print(f"  [verdict] cannot judge - baseline has {len(scores)} completed seeds, need 3.")
        print("            Showing the gaps as numbers only. A threshold invented from")
        print("            nothing would pass anything and only look like a judgement.")
        return

    base = statistics.mean(scores)
    print(f"  baseline '{baseline_arm}': mean={base:.5f} sigma={statistics.stdev(scores):.5f} "
          f"(n={len(scores)})")

    for arm, rows in sorted(arms.items()):
        if arm == baseline_arm:
            continue
        values = [r["map"] for r in rows if isinstance(r.get("map"), (int, float))]
        if not values:
            continue
        gain = statistics.mean(values) - base
        limit = threshold(scores, values)
        if gain >= limit:
            verdict = "SURVIVES  (beats noise)"
        elif gain <= -limit:
            verdict = "HARMFUL   (loses beyond noise)"
        else:
            verdict = "INDISTINGUISHABLE from noise"
        print(f"  {arm:12s} n={len(values)} delta={gain:+.5f} threshold=+-{limit:.5f}  {verdict}")


def main(argv: list[str]) -> int:
    if not DB_PATH.is_file():
        print(f"[error] no database at {DB_PATH}")
        return 1

    blocks = collect()
    if not blocks:
        print("no completed p6/ runs yet.")
        return 0

    wanted = [a.upper() for a in argv if not a.startswith("-")] or sorted(blocks)
    for block in wanted:
        if block not in blocks:
            print(f"=== block {block}: no completed runs yet")
            continue
        show_measurements(block, blocks[block])
        show_verdict(block, blocks[block])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
