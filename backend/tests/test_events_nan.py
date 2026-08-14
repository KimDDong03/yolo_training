"""events.jsonl 이 브라우저가 읽을 수 있는 JSON 만 담는지 검증한다.

파이썬 json 은 NaN 을 bare `NaN` 리터럴로 쓰고 또 읽지만, 브라우저 JSON.parse 는 거기서
SyntaxError 로 죽는다. 그러면 loss 가 발산하는 바로 그 순간 실시간 화면이 통째로 멎는다.
`parse_constant` 로 그 엄격한 파서를 흉내 내 회귀를 잡는다.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from . import _support  # noqa: F401  (sys.path 설정)

HOOKS_DIR = Path(__file__).resolve().parents[1] / "hooks"
if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))

NAN = float("nan")
INF = float("inf")


def strict_loads(text: str):
    """브라우저 JSON.parse 와 같게 NaN/Infinity 를 거부하는 파서."""

    def reject(constant: str):
        raise ValueError(f"JSON.parse 가 거부하는 토큰: {constant}")

    return json.loads(text, parse_constant=reject)


class NumConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        import yoloweb_events

        self.ev = yoloweb_events

    def test_nonfinite_becomes_none(self) -> None:
        for value in (NAN, INF, -INF):
            self.assertIsNone(self.ev._num(value))

    def test_ordinary_values_pass_through(self) -> None:
        self.assertEqual(self.ev._num(1.5), 1.5)
        self.assertEqual(self.ev._num(7), 7)
        self.assertEqual(self.ev._num("x"), "x")
        self.assertIs(self.ev._num(True), True)
        self.assertIsNone(self.ev._num(None))

    def test_nonfinite_keys_lists_what_was_dropped(self) -> None:
        keys = self.ev._nonfinite_keys({"a": NAN, "b": 1.0, "c": INF, "d": None})
        self.assertEqual(keys, ["a", "c"])

    def test_cuda_mem_is_none_on_cpu_run(self) -> None:
        """GPU 가 꽂힌 PC 에서 CPU 로 학습해도 VRAM 을 0.0 으로 기록하면 안 된다.

        그 0.0 이 나중에 VRAM 추정의 실측 표본으로 섞이면 추정이 통째로 어긋난다.
        """

        class Trainer:
            device = "cpu"

        self.assertIsNone(self.ev._cuda_mem_gb(Trainer()))
        self.assertIsNone(self.ev._cuda_mem_gb(object()))


class WriterOutputTest(unittest.TestCase):
    """실제 콜백을 태워 파일에 무엇이 남는지 본다 (헬퍼 단위가 아니라 통합)."""

    def setUp(self) -> None:
        self.run_dir = Path(tempfile.mkdtemp(prefix="events_"))
        os.environ["YOLOWEB_RUN_DIR"] = str(self.run_dir)
        import yoloweb_events

        self.ev = yoloweb_events
        self.ev.install()

        class Trainer:
            device = "cpu"
            epoch, epochs = 0, 3
            loss = NAN
            tloss = None
            metrics = {"metrics/mAP50(B)": NAN, "metrics/mAP50-95(B)": 0.31}
            lr = {"pg0": 0.01}
            fitness, best_fitness, epoch_time = NAN, 0.4, 1.25

            def label_loss_items(self, _):
                return {"train/box_loss": NAN}

        self.trainer = Trainer()

    def _events(self) -> list[dict]:
        raw = (self.run_dir / "events.jsonl").read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        for line in lines:
            self.assertNotIn("NaN", line, "파일에 bare NaN 리터럴이 남았다")
            self.assertNotIn("Infinity", line)
        return [strict_loads(line) for line in lines]

    def test_batch_loss_nan_is_flagged_not_written(self) -> None:
        self.ev.on_train_batch_end(self.trainer)
        event = self._events()[0]
        self.assertEqual(event["t"], "batch")
        self.assertIsNone(event["loss"])
        self.assertTrue(event["loss_nan"])

    def test_epoch_keeps_good_metrics_and_names_the_bad(self) -> None:
        self.ev.on_fit_epoch_end(self.trainer)
        event = self._events()[0]
        # 좋은 값은 살아남고 NaN 인 것만 사라진다 — 에폭 전체를 버리지 않는다.
        self.assertEqual(event["summary"]["mAP50-95"], 0.31)
        self.assertIsNone(event["summary"]["mAP50"])
        self.assertIn("metrics/mAP50(B)", event["nonfinite"])
        self.assertIn("train/box_loss", event["nonfinite"])
        self.assertEqual(event["epoch_time_s"], 1.25)
        self.assertIsNone(event["mem_gb"])


class JsonSafeTest(unittest.TestCase):
    """이 수정 이전에 만들어진 events.jsonl 이 디스크에 남아 있다."""

    def setUp(self) -> None:
        from app.services.event_stream import KEEP_KINDS, json_safe

        self.json_safe = json_safe
        self.keep_kinds = KEEP_KINDS

    def test_legacy_nan_file_becomes_parseable(self) -> None:
        legacy = json.loads('{"t":"epoch","metrics":{"loss":NaN,"map":0.5},"xs":[NaN,1]}')
        cleaned = self.json_safe(legacy)
        strict_loads(json.dumps(cleaned))  # 던지면 실패
        self.assertIsNone(cleaned["metrics"]["loss"])
        self.assertEqual(cleaned["metrics"]["map"], 0.5)
        self.assertEqual(cleaned["xs"], [None, 1])

    def test_warning_survives_snapshot(self) -> None:
        """빠지면 이상 감지 배지가 라이브에서만 보이고 새로고침하면 사라진다."""
        self.assertIn("warning", self.keep_kinds)


if __name__ == "__main__":
    unittest.main()
