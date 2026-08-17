"""Phase 6 실측 캠페인의 학습을 블록 단위로 큐에 넣는다.

    python scripts/campaign_enqueue.py A        # 블록 하나
    python scripts/campaign_enqueue.py --list   # 계획만 보여주고 넣지 않는다

앱의 기존 큐를 그대로 쓴다. POST /api/runs 가 queued 로 넣고 run_manager.schedule() 이
GPU 가 비면 하나씩 꺼내므로, GPU 1장에서는 저절로 순차 실행된다. 별도 스케줄러가 없다.

**블록 단위로 넣는 이유:** 전부 한 번에 넣으면 앞 블록의 결과를 보고 뒷 블록을 조정할
기회가 사라진다. 블록 A 의 VRAM 실측이 블록 G 의 batch 를 정하고, 어느 규칙이 발화하는지가
어느 블록을 도는지를 정한다.

**멱등하다.** 같은 이름의 run 이 이미 completed 면 건너뛴다. failed/stopped 는 다시 넣는다 —
이름만 보고 건너뛰면 정확히 다시 돌려야 할 실패 run 을 건너뛰게 된다.

venv 활성화 전에도 돌아야 해서 표준 라이브러리만 쓴다. 콘솔이 cp949 라 출력은 ASCII 로 한다.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"

# 데이터셋 이름 -> 학습에 쓸 train 장수. 시간 예측에만 쓴다.
TRAIN_COUNT = {
    "brain-tumor": 893,
    "african-wildlife": 1052,
    "HomeObjects-3K": 2285,
    "medical-pills": 92,
    "signature": 143,
}

# 실측 상수. run 3개(brain-tumor/african-wildlife/HomeObjects)에서 초/(장·에폭)이
# 0.01067 / 0.01006 / 0.01051 로 +-3% 안에 들어왔다. 고정항은 최소자승 절편이 0.24초라 뺐다.
SEC_PER_IMAGE_EPOCH = 0.0104
# 프로세스 기동 + torch/ultralytics import + 데이터 스캔 + AMP 체크.
RUN_OVERHEAD_S = 60.0
# yolo11n 을 1.0 으로 둔 상대 연산량. **이번 블록 A 가 재는 대상이라 아직 추측이다.**
MODEL_COST = {"n": 1.0, "s": 1.9, "m": 4.3}


def scale_of(model: str) -> str:
    stem = model.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    for s in ("n", "s", "m", "l", "x"):
        if stem.startswith(f"yolo11{s}"):
            return s
    return "n"


def predict_seconds(dataset: str, params: dict) -> float:
    images = TRAIN_COUNT.get(dataset, 1000)
    imgsz = int(params.get("imgsz", 640))
    epochs = int(params.get("epochs", 100))
    cost = MODEL_COST.get(scale_of(str(params.get("model", ""))), 1.0)
    per_epoch = SEC_PER_IMAGE_EPOCH * images * (imgsz / 640.0) ** 2 * cost
    return RUN_OVERHEAD_S + epochs * max(2.0, per_epoch)


def api(path: str, payload: dict | None = None) -> object:
    url = BASE + path
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


# --------------------------------------------------------------- 블록 정의
#
# arm = (이름, 파라미터 덮어쓰기, 시드들). 파라미터는 폼 기본값 위에 얹는다 —
# 사용자가 실제로 받는 값과 같아야 여기서 나온 상수가 앱 안에서 유효하다.

# 블록 A·C 를 뺀 모든 블록은 batch 를 명시적으로 고정한다.
# 프리셋의 batch=-1(AutoBatch)은 그때 비어 있는 VRAM 을 보고 정하므로, 시드만 바꿔도
# batch 가 달라질 수 있다. 그러면 표준편차가 시드 노이즈가 아니라 batch 변동을 재게 된다.
FIXED_BATCH = 16

BLOCKS: dict[str, dict] = {
    # 시간·VRAM 상수를 한 번에 푼다. batch 2점이 있어야 mem = BASE + PER_IMAGE*batch 가
    # 기울기와 절편으로 분해된다. b4 를 먼저 돌려 m 의 기울기를 배운 뒤 b32 를 건다.
    "A": {
        "dataset": "brain-tumor",
        "note": "time/VRAM probe - 3 epochs, batch 4 then 32, n/s/m",
        "arms": [
            (f"{m}-b{b}", {"model": f"yolo11{m}.pt", "batch": b, "epochs": 3}, [0])
            for b in (4, 32)
            for m in ("n", "s", "m")
        ],
    },
    # 블록 A 의 두 batch 는 **둘 다 MODEL_COST 산출에 못 쓴다.**
    # b4 는 GPU 가 놀아 연산량이 아니라 오버헤드를 재고(s 19.71초가 n 20.77초보다 빨랐다),
    # b32 는 m 이 16.4GB 를 잡아 Windows 시스템 메모리로 흘러 4.7배 느려졌다.
    # 그래서 세 스케일이 전부 VRAM 안에 들어오면서 GPU 를 채우는 batch 16 에서 다시 잰다.
    "A2": {
        "dataset": "brain-tumor",
        "note": "clean MODEL_COST at batch 16 - b4 idles the GPU, b32 spills m to host memory",
        "arms": [
            (f"{m}-b16", {"model": f"yolo11{m}.pt", "batch": 16, "epochs": 3}, [0])
            for m in ("n", "s", "m")
        ],
    },
    # 프리셋. epochs 를 100 으로 묶고 축을 하나씩 켠다. 5축을 한 번에 재면 어느 축이
    # 기여했는지 영영 모른다. headline 을 위해 고정밀 출고본도 한 arm 남긴다.
    # 이 블록만 batch -1 을 그대로 둔다 - 프리셋 출고 상태를 재는 것이 질문이기 때문이다.
    "F": {
        "dataset": "brain-tumor",
        "note": "preset factorial - balanced baseline (3 seeds) + one axis each",
        "arms": [
            (
                "balanced",
                {"epochs": 100, "imgsz": 640, "batch": -1, "patience": 30},
                [0, 1, 2],
            ),
            (
                "imgsz800",
                {"epochs": 100, "imgsz": 800, "batch": -1, "patience": 30},
                [0],
            ),
            (
                "cos_lr",
                {
                    "epochs": 100,
                    "imgsz": 640,
                    "batch": -1,
                    "patience": 30,
                    "cos_lr": True,
                },
                [0],
            ),
            (
                "cm20",
                {
                    "epochs": 100,
                    "imgsz": 640,
                    "batch": -1,
                    "patience": 30,
                    "close_mosaic": 20,
                },
                [0],
            ),
            (
                "precise",
                {
                    "epochs": 300,
                    "imgsz": 800,
                    "batch": -1,
                    "patience": 50,
                    "cos_lr": True,
                    "close_mosaic": 20,
                },
                [0],
            ),
        ],
    },
    # 모델 크기. n arm 이 이 블록의 기준이자 표준편차 표본이다.
    "G": {
        "dataset": "brain-tumor",
        "note": "model size n/s/m - 30 epochs, 3 seeds each",
        "arms": [
            (
                f"yolo11{m}",
                {"model": f"yolo11{m}.pt", "epochs": 30, "batch": FIXED_BATCH},
                [0, 1, 2],
            )
            for m in ("n", "s", "m")
        ],
    },
    # large_objects 규칙. african-wildlife 에서만 발화한다(median_area 0.2131).
    # patch 는 손으로 쓰지 않고 추천 API 가 실제로 낸 {"imgsz": 480} 그대로다.
    "B": {
        "dataset": "african-wildlife",
        "note": "large_objects rule - imgsz 640 vs 480",
        "arms": [
            ("off640", {"imgsz": 640, "epochs": 30, "batch": FIXED_BATCH}, [0, 1, 2]),
            ("on480", {"imgsz": 480, "epochs": 30, "batch": FIXED_BATCH}, [0, 1, 2]),
        ],
    },
    # few_images 규칙. 추천 API 가 낸 patch 는 {"epochs": 300, "mixup": 0.1} 이다 -
    # patience 50 은 들어 있지 않다(폼 기본 patience 가 100 이라 그 가지가 죽어 있다).
    "D": {
        "dataset": "medical-pills",
        "note": "few_images rule - 92 images, epochs 100 vs 300+mixup",
        "arms": [
            ("off", {"epochs": 100, "batch": FIXED_BATCH}, [0, 1, 2]),
            ("on", {"epochs": 300, "mixup": 0.1, "batch": FIXED_BATCH}, [0, 1, 2]),
        ],
    },
    # many_images 규칙의 잴 수 있는 절반. cache=disk 는 mAP 가 아니라 처리량 주장이라
    # 3에폭이면 끝난다. 캐시 빌드 비용과 정상 상태를 분리하려고 설정마다 2회 돈다.
    "C": {
        "dataset": "HomeObjects-3K",
        "note": "cache=disk throughput only - not a mAP claim",
        "arms": [
            ("nocache", {"epochs": 3, "batch": FIXED_BATCH, "cache": "False"}, [0, 1]),
            ("disk", {"epochs": 3, "batch": FIXED_BATCH, "cache": "disk"}, [0, 1]),
        ],
    },
    # few_images 복제. 표본 1개로 규칙을 죽이거나 살리지 않기 위한 두 번째 데이터셋.
    "H": {
        "dataset": "signature",
        "note": "few_images replicate on a second dataset",
        "arms": [
            ("off", {"epochs": 100, "batch": FIXED_BATCH}, [0, 1, 2]),
            ("on", {"epochs": 300, "mixup": 0.1, "batch": FIXED_BATCH}, [0]),
        ],
    },
}

ORDER = ["A", "A2", "F", "G", "B", "D", "C", "H"]


def plan(block: str) -> list[tuple[str, str, dict]]:
    """이 블록이 만들 (run 이름, 데이터셋 이름, 파라미터) 목록."""
    spec = BLOCKS[block]
    rows = []
    for arm, overrides, seeds in spec["arms"]:
        for seed in seeds:
            rows.append(
                (
                    f"p6/{block}/{arm}/s{seed}",
                    spec["dataset"],
                    {**overrides, "seed": seed},
                )
            )
    return rows


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("blocks:", " ".join(ORDER))
        return 0

    show_only = "--list" in argv
    blocks = [a.upper() for a in argv if not a.startswith("-")]
    if not blocks:
        blocks = ORDER
    unknown = [b for b in blocks if b not in BLOCKS]
    if unknown:
        print(f"[error] unknown block: {unknown}. known: {ORDER}")
        return 2

    try:
        schema = api("/api/params/schema")
        datasets = api("/api/datasets")
        existing = api("/api/runs")
    except (urllib.error.URLError, OSError) as exc:
        print(f"[error] app not reachable at {BASE}: {exc}")
        print("        start it with: python backend/run.py")
        return 1

    assert (
        isinstance(schema, dict)
        and isinstance(datasets, list)
        and isinstance(existing, list)
    )
    defaults = {
        f["key"]: f["default"]
        for f in schema["schema"]["fields"]
        if f.get("scope") == "params"
    }
    ids = {d["name"]: d["id"] for d in datasets}
    done = {r["name"] for r in existing if r.get("status") == "completed"}
    live = {r["name"] for r in existing if r.get("status") in ("queued", "running")}

    total_seconds = 0.0
    queued = skipped = failed = 0

    for block in blocks:
        print(f"=== block {block}: {BLOCKS[block]['note']}")
        for name, dataset_name, overrides in plan(block):
            params = {**defaults, **overrides}
            seconds = predict_seconds(dataset_name, params)

            if name in done:
                print(f"  [skip]  {name}  (completed)")
                skipped += 1
                continue
            if name in live:
                print(f"  [skip]  {name}  (already queued/running)")
                skipped += 1
                continue
            if dataset_name not in ids:
                print(f"  [error] {name}  dataset not registered: {dataset_name}")
                failed += 1
                continue

            total_seconds += seconds
            if show_only:
                print(f"  [plan]  {name}  ~{seconds / 60:.1f} min")
                continue

            try:
                api(
                    "/api/runs",
                    {
                        "dataset_id": ids[dataset_name],
                        "name": name,
                        "params": params,
                        "devices": [0],
                    },
                )
                print(f"  [queue] {name}  ~{seconds / 60:.1f} min")
                queued += 1
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:200]
                print(f"  [error] {name}  HTTP {exc.code}: {body}")
                failed += 1

    print()
    print(
        f"queued={queued} skipped={skipped} failed={failed}"
        f"  predicted GPU time {total_seconds / 3600:.2f} h"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
