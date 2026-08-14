"""학습이 도는 중에 "지금 잘 되고 있는가" 를 대신 판단한다.

차트를 그려 주는 것만으로는 부족하다. 손실이 발산했는지, 몇 시간째 성능이 제자리인지,
과적합이 시작됐는지는 곡선을 읽을 줄 알아야 보인다. 비전문가는 몇 시간을 태우고 나서야
알게 된다. 그래서 백엔드가 판정해 경고를 남긴다.

판정을 프론트가 아니라 여기서 하는 이유:
  - 데이터로더 병목 판정에 필요한 GPU 사용률이 프론트에 없다. 서버 시각과 브라우저
    수신 시각을 조인해야 한다.
  - useRunStream 은 이미 스냅샷 교체·중복 흡수·백오프 재연결을 다룬다. 여기에 통계 판정을
    얹으면 재연결마다 전량 재계산이 붙는다.
  - 기준을 고치려면 프론트는 frontend/dist 를 다시 반입해야 하지만 백엔드는 .py 하나다.

전달은 events.jsonl 에 warning 을 append 하는 것으로 끝난다. 새 API 도 새 폴링도 없고,
파일이 단일 원천이라 새로고침·재접속 복원이 공짜로 따라온다. 백엔드가 이 파일에 쓰는 것은
이미 있는 관례다(run_manager 의 _preflight 와 _settle). 지켜야 할 불변식은 "한 줄 = 한 번의
write" 뿐이다.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from app.core import db
from app.services import gpu, run_manager
from app.services.event_stream import _Tailer

# 성능이 이만큼도 안 오르면 개선으로 치지 않는다.
MIN_IMPROVEMENT = 0.001
# 정체 판정에 필요한 최소 에폭 수 (patience 가 0일 때).
STALL_EPOCHS = 5
# 과적합은 초반 변동이 큰 구간을 지나서 본다.
OVERFIT_MIN_EPOCH = 8
OVERFIT_RISING = 3
# 배치 이벤트가 이만큼 끊기면 멈춘 것으로 본다. 검증·캐시 빌드로도 끊기므로 넉넉히 잡는다.
STALL_SECONDS = 180.0
# 데이터로더 병목 판정
GPU_SAMPLE_WINDOW_S = 30.0
GPU_IDLE_THRESHOLD = 35.0

_watchers: dict[str, "_Watcher"] = {}


def _loss_sum(metrics: dict[str, Any], prefix: str) -> float | None:
    values = [
        v
        for k, v in metrics.items()
        if k.startswith(prefix) and k.endswith("_loss") and isinstance(v, (int, float))
    ]
    if not values or any(not math.isfinite(v) for v in values):
        return None
    return float(sum(values))


class _Watcher:
    """run 하나의 누적 상태. 오프셋 기반으로 새 줄만 읽는다.

    매번 events.jsonl 을 처음부터 읽으면, 배치 이벤트가 수만 줄 쌓인 긴 학습에서
    1초마다 수 MB 를 다시 파싱하게 된다.
    """

    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.tailer = _Tailer(run_dir / "events.jsonl")
        self.emitted: set[str] = set()
        self.epochs: list[dict[str, Any]] = []
        self.best_fitness: float | None = None
        self.best_epoch = 0
        self.last_batch_at: float | None = None
        self.saw_nan = False
        self.gpu_samples: list[tuple[float, float]] = []

    def ingest(self) -> None:
        for line in self.tailer.read_lines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("t")

            if kind == "warning":
                # 이미 낸 경고는 다시 내지 않는다. 파일에서 읽어 복원하므로
                # 백엔드를 재시작해도 같은 경고가 두 번 붙지 않는다.
                code = event.get("code")
                if isinstance(code, str):
                    self.emitted.add(code)
            elif kind == "batch":
                self.last_batch_at = time.time()
                if event.get("loss_nan"):
                    self.saw_nan = True
            elif kind == "epoch":
                metrics = event.get("metrics") or {}
                if event.get("nonfinite"):
                    self.saw_nan = True
                fitness = event.get("fitness")
                epoch = int(event.get("epoch") or 0)
                if isinstance(fitness, (int, float)) and math.isfinite(fitness):
                    if (
                        self.best_fitness is None
                        or fitness > self.best_fitness + MIN_IMPROVEMENT
                    ):
                        self.best_fitness = float(fitness)
                        self.best_epoch = epoch
                self.epochs.append(
                    {
                        "epoch": epoch,
                        "train": _loss_sum(metrics, "train/"),
                        "val": _loss_sum(metrics, "val/"),
                    }
                )

    def sample_gpu(self, devices: list[int]) -> None:
        if not devices:
            return
        now = time.time()
        for entry in gpu.list_gpus():
            if int(entry["index"]) in devices:  # type: ignore[arg-type]
                self.gpu_samples.append((now, float(entry["utilization"])))  # type: ignore[arg-type]
                break
        cutoff = now - GPU_SAMPLE_WINDOW_S
        self.gpu_samples = [s for s in self.gpu_samples if s[0] >= cutoff]


def _rules(watcher: _Watcher, run: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    epochs = watcher.epochs
    latest = epochs[-1]["epoch"] if epochs else 0

    if watcher.saw_nan:
        found.append(
            {
                "code": "loss_nan",
                "severity": "critical",
                "message": "손실이 NaN 으로 발산했습니다. 이대로 두면 아무것도 학습되지 않습니다.",
                "hint": "안전 정지 후 학습률을 낮추거나 AMP(혼합정밀)를 꺼서 다시 시작하세요.",
            }
        )

    patience = int(run["params"].get("patience", 0) or 0)
    window = max(STALL_EPOCHS, patience // 3) if patience else STALL_EPOCHS
    if latest and watcher.best_epoch and latest - watcher.best_epoch >= window:
        stalled_for = latest - watcher.best_epoch
        hint = (
            f"조기 종료(patience {patience})가 곧 작동합니다. 그대로 두어도 됩니다."
            if patience
            else "조기 종료가 꺼져 있어 끝까지 돕니다. 정지하고 설정을 바꾸는 편이 나을 수 있습니다."
        )
        found.append(
            {
                "code": "map_stall",
                "severity": "info",
                "message": f"{stalled_for}에폭째 최고 성능이 갱신되지 않았습니다 "
                f"({watcher.best_epoch}에폭의 {watcher.best_fitness:.4f} 이 최고).",
                "hint": hint,
            }
        )

    # 과적합: 검증 손실은 오르는데 학습 손실은 내려가는 구간
    if latest >= OVERFIT_MIN_EPOCH and len(epochs) > OVERFIT_RISING:
        tail = epochs[-(OVERFIT_RISING + 1) :]
        vals = [e["val"] for e in tail]
        trains = [e["train"] for e in tail]
        if all(v is not None for v in vals) and all(t is not None for t in trains):
            val_rising = all(vals[i + 1] > vals[i] for i in range(len(vals) - 1))  # type: ignore[operator]
            train_falling = trains[-1] < trains[0]  # type: ignore[operator]
            if val_rising and train_falling:
                found.append(
                    {
                        "code": "overfit",
                        "severity": "warn",
                        "message": f"검증 손실이 {OVERFIT_RISING}에폭 연속 오르는데 학습 손실은 내려갑니다 "
                        f"(과적합 신호).",
                        "hint": "조기 종료(patience)를 켜거나 증강을 강화하고, 데이터를 늘리는 것이 근본 대책입니다.",
                    }
                )

    # 데이터로더 병목: GPU 는 노는데 배치는 돌아가는 중
    samples = watcher.gpu_samples
    if (
        run["devices"]
        and len(run["devices"]) == 1  # DDP 는 사용률 해석이 다르다
        and latest >= 1  # 첫 에폭 전에는 준비 단계라 낮은 게 정상이다
        and len(samples) >= 5
        and samples[-1][0] - samples[0][0] >= GPU_SAMPLE_WINDOW_S * 0.5
    ):
        average = sum(u for _, u in samples) / len(samples)
        if average < GPU_IDLE_THRESHOLD:
            found.append(
                {
                    "code": "dataloader_slow",
                    "severity": "warn",
                    "message": f"GPU 사용률이 최근 평균 {average:.0f}% 로 낮습니다. "
                    f"계산이 아니라 이미지를 읽는 데서 막히고 있을 가능성이 큽니다.",
                    "hint": "데이터 로더 워커를 늘리거나 이미지 캐시(cache)를 켜 보세요.",
                }
            )

    if watcher.last_batch_at and time.time() - watcher.last_batch_at > STALL_SECONDS:
        found.append(
            {
                "code": "stalled",
                "severity": "info",
                "message": f"{int(time.time() - watcher.last_batch_at)}초째 배치 진행이 보이지 않습니다.",
                "hint": "검증이나 이미지 캐시 생성 중일 수 있습니다. 더 길어지면 로그를 확인하세요.",
            }
        )

    return found


def scan() -> None:
    """진행 중인 학습을 훑어 새로 발견한 이상을 events.jsonl 에 남긴다.

    스케줄러 루프에서 매초 호출된다. 어떤 예외도 학습 스케줄링을 막으면 안 되므로
    run 하나가 실패해도 나머지는 계속 본다.
    """
    running = {
        row["id"]: db.row_to_run(row)
        for row in db.query("SELECT * FROM runs WHERE status = 'running'")
    }

    for run_id in list(_watchers):
        if run_id not in running:
            _watchers.pop(run_id, None)

    for run_id, run in running.items():
        try:
            watcher = _watchers.get(run_id)
            if watcher is None:
                watcher = _Watcher(run_id, run_manager.run_dir_for(run_id))
                _watchers[run_id] = watcher
            watcher.ingest()
            watcher.sample_gpu(run["devices"])

            for warning in _rules(watcher, run):
                if warning["code"] in watcher.emitted:
                    continue
                watcher.emitted.add(warning["code"])
                run_manager._append_event(run_id, {"t": "warning", **warning})
        except Exception:  # noqa: BLE001 - 감시가 스케줄러를 죽여선 안 된다
            import traceback

            traceback.print_exc()
