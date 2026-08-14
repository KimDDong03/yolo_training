"""학습 중 이상 감지 검증.

경고는 한 번만 떠야 하고, 백엔드를 재시작해도 다시 뜨면 안 된다. 그 두 가지가
이 기능에서 제일 틀리기 쉬운 부분이라 집중해서 고정한다.
"""

from __future__ import annotations

import json
import time
import unittest

from ._support import isolate_storage


class AnomalyTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = isolate_storage()
        from app.core import config, db
        from app.services import anomaly

        self.config, self.db, self.anomaly = config, db, anomaly
        anomaly._watchers.clear()
        self.run_id = "run1"
        self.run_dir = config.RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = self.run_dir / "events.jsonl"
        self.events.write_text("", encoding="utf-8")

    def make_run(self, *, params: dict | None = None, devices: list[int] | None = None) -> None:
        params = params or {"patience": 0}
        self.db.execute(
            "INSERT INTO runs"
            " (id,name,dataset_id,status,params,options,devices,retry_of,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (self.run_id, "t", "ds", "running", json.dumps(params), "{}",
             json.dumps(devices if devices is not None else [0]), None, time.time()),
        )

    def append(self, event: dict) -> None:
        with open(self.events, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def epoch(self, n: int, *, fitness: float, train: float = 1.0, val: float = 1.0, **extra) -> None:
        self.append({
            "t": "epoch",
            "epoch": n,
            "fitness": fitness,
            "metrics": {
                "train/box_loss": train, "train/cls_loss": 0.0, "train/dfl_loss": 0.0,
                "val/box_loss": val, "val/cls_loss": 0.0, "val/dfl_loss": 0.0,
            },
            **extra,
        })

    def warnings(self) -> list[dict]:
        out = []
        for line in self.events.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("t") == "warning":
                out.append(event)
        return out

    def codes(self) -> list[str]:
        return [w["code"] for w in self.warnings()]


class DetectionTest(AnomalyTestBase):
    def test_nan_batch_raises_a_critical_warning(self) -> None:
        self.make_run()
        self.append({"t": "batch", "epoch": 1, "loss": None, "loss_nan": True})
        self.anomaly.scan()
        warnings = self.warnings()
        self.assertEqual([w["code"] for w in warnings], ["loss_nan"])
        self.assertEqual(warnings[0]["severity"], "critical")
        self.assertTrue(warnings[0]["hint"])

    def test_nan_metrics_in_epoch_also_count(self) -> None:
        self.make_run()
        self.epoch(1, fitness=0.1, nonfinite=["train/box_loss"])
        self.anomaly.scan()
        self.assertIn("loss_nan", self.codes())

    def test_stalled_progress_is_reported(self) -> None:
        self.make_run(params={"patience": 0})
        self.epoch(1, fitness=0.90)
        for n in range(2, 9):
            self.epoch(n, fitness=0.90)  # 개선 없음
        self.anomaly.scan()
        self.assertIn("map_stall", self.codes())

    def test_steady_improvement_is_not_flagged(self) -> None:
        self.make_run()
        for n in range(1, 12):
            self.epoch(n, fitness=0.1 * n)
        self.anomaly.scan()
        self.assertNotIn("map_stall", self.codes())

    def test_overfitting_is_detected(self) -> None:
        """검증 손실은 오르는데 학습 손실은 내려가는 구간."""
        self.make_run()
        for n in range(1, 9):
            self.epoch(n, fitness=0.5, train=2.0 - n * 0.1, val=1.0)
        self.epoch(9, fitness=0.5, train=1.0, val=1.1)
        self.epoch(10, fitness=0.5, train=0.9, val=1.2)
        self.epoch(11, fitness=0.5, train=0.8, val=1.3)
        self.anomaly.scan()
        self.assertIn("overfit", self.codes())

    def test_early_epochs_are_not_called_overfitting(self) -> None:
        """초반에는 검증 손실이 흔들린다. 그걸 과적합이라고 하면 매번 뜬다."""
        self.make_run()
        for n, val in [(1, 1.0), (2, 1.1), (3, 1.2), (4, 1.3)]:
            self.epoch(n, fitness=0.5, train=2.0 - n * 0.1, val=val)
        self.anomaly.scan()
        self.assertNotIn("overfit", self.codes())

    def test_both_losses_rising_is_not_overfitting(self) -> None:
        """둘 다 오르면 그냥 학습이 안 되는 것이지 과적합이 아니다."""
        self.make_run()
        for n in range(1, 12):
            self.epoch(n, fitness=0.5, train=1.0 + n * 0.1, val=1.0 + n * 0.1)
        self.anomaly.scan()
        self.assertNotIn("overfit", self.codes())

    def test_cpu_runs_skip_the_gpu_rule(self) -> None:
        self.make_run(devices=[])
        for n in range(1, 4):
            self.epoch(n, fitness=0.1 * n)
        for _ in range(10):
            self.anomaly.scan()
        self.assertNotIn("dataloader_slow", self.codes())


class DeduplicationTest(AnomalyTestBase):
    def test_the_same_warning_is_emitted_only_once(self) -> None:
        self.make_run()
        self.append({"t": "batch", "epoch": 1, "loss_nan": True})
        for _ in range(5):
            self.anomaly.scan()
        self.assertEqual(self.codes().count("loss_nan"), 1)

    def test_restart_does_not_repeat_past_warnings(self) -> None:
        """감시자 상태는 메모리에 있지만 파일이 단일 원천이다.

        재시작 후 기존 warning 을 읽어 복원하지 않으면 같은 경고가 다시 붙는다.
        """
        self.make_run()
        self.append({"t": "batch", "epoch": 1, "loss_nan": True})
        self.anomaly.scan()
        self.assertEqual(self.codes().count("loss_nan"), 1)

        self.anomaly._watchers.clear()  # 백엔드 재시작
        self.anomaly.scan()
        self.assertEqual(self.codes().count("loss_nan"), 1)

    def test_finished_runs_are_dropped_from_memory(self) -> None:
        self.make_run()
        self.append({"t": "batch", "epoch": 1, "loss_nan": True})
        self.anomaly.scan()
        self.assertIn(self.run_id, self.anomaly._watchers)

        self.db.execute("UPDATE runs SET status='completed' WHERE id=?", (self.run_id,))
        self.anomaly.scan()
        self.assertNotIn(self.run_id, self.anomaly._watchers)


class IncrementalReadTest(AnomalyTestBase):
    def test_events_are_read_incrementally(self) -> None:
        """매 스캔마다 파일을 처음부터 읽으면 긴 학습에서 1초마다 수 MB 를 재파싱한다."""
        self.make_run()
        for n in range(1, 4):
            self.epoch(n, fitness=0.1 * n)
        self.anomaly.scan()
        watcher = self.anomaly._watchers[self.run_id]
        offset = watcher.tailer.offset
        self.assertGreater(offset, 0)
        self.assertEqual(len(watcher.epochs), 3)

        self.anomaly.scan()  # 새 줄이 없다
        self.assertEqual(watcher.tailer.offset, offset)
        self.assertEqual(len(watcher.epochs), 3, "같은 줄을 다시 읽었다")

        self.epoch(4, fitness=0.4)
        self.anomaly.scan()
        self.assertEqual(len(watcher.epochs), 4)

    def test_warnings_are_valid_json_lines(self) -> None:
        """이 파일은 브라우저가 읽는다. 한 줄이라도 깨지면 스트림이 멎는다."""
        self.make_run()
        self.append({"t": "batch", "epoch": 1, "loss_nan": True})
        self.anomaly.scan()
        for line in self.events.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


if __name__ == "__main__":
    unittest.main()
