"""시간 예산(`time`) 파라미터.

ultralytics 8.4.47 이 `time`(시간 단위)을 받으면 에폭 수를 무시하고 wall-clock 으로
자른다. 매 에폭 끝에서 남은 예산으로 에폭 수를 다시 계산하고 LR 스케줄러도 다시 만든다
(engine/trainer.py:546-547).

여기서 고정하는 것은 두 가지다.

1. 폼 스키마가 이 필드를 내려주고 기본값이 `None` 이 아닌 `0.0` 이라는 것.
   ultralytics 기본값이 None 이라 보정하지 않으면 폼이 null 을 렌더한다.
2. 예상 시간이 예산을 반영한다는 것. 안 그러면 30분 예산 실행에 4시간이라고 답한다.

**엄격한 상한이 아니다.** 시간 검사가 배치 단위로만 돌고(trainer.py:474) 걸린 에폭의
검증과 마지막 정리가 뒤에 더 붙는다. 그래서 "실제 소요가 예산을 넘지 않는다" 는
주장하지 않는다 - 그건 실측으로 초과 폭을 재서 다룬다.
"""

from __future__ import annotations

import unittest

from ._support import isolate_storage


def dataset(train: int = 5000) -> dict:
    return {
        "report": {
            "train_count": train,
            "total_images": train,
            "class_instances": {"a": 100},
            "issue_counts": {},
        }
    }


class TimeBudgetSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        from app.services import param_schema

        self.ps = param_schema

    def test_schema_exposes_the_field(self) -> None:
        field = self.ps.field_index().get("time")
        self.assertIsNotNone(field)
        assert field is not None
        self.assertEqual(field["type"], "float")
        self.assertEqual(field["scope"], "params")
        # 전문 모드에만 있으면 간편 모드 사용자가 못 본다. 그 사용자가 대상이다.
        self.assertFalse(field["advanced"])

    def test_default_is_zero_not_none(self) -> None:
        """ultralytics 기본값은 None 이다. 그대로 내보내면 폼이 null 을 렌더한다."""
        self.assertEqual(self.ps.field_index()["time"]["default"], 0.0)
        self.assertEqual(self.ps.defaults_dict("params")["time"], 0.0)

    def test_range_is_enforced(self) -> None:
        self.assertEqual(self.ps.validate({"time": 0.5}, "params"), {"time": 0.5})
        self.assertEqual(self.ps.validate({"time": 0}, "params"), {"time": 0.0})
        for bad in (99, -1):
            with self.subTest(value=bad):
                with self.assertRaises(self.ps.ValidationError):
                    self.ps.validate({"time": bad}, "params")

    def test_default_form_still_passes_the_allowlist(self) -> None:
        """필드를 늘렸으니 기본값 전체가 그대로 관문을 통과하는지 다시 확인한다."""
        defaults = self.ps.defaults_dict("params")
        self.assertEqual(self.ps.validate(defaults, "params")["time"], 0.0)


class TimeBudgetEstimateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = isolate_storage()
        from app.services import estimate, param_schema

        self.estimate = estimate
        self.defaults = param_schema.defaults_dict("params")

    def form(self, **kw) -> dict:
        return {**self.defaults, "model": "yolo11n.pt", "imgsz": 640, "batch": 16,
                "epochs": 100, **kw}

    def test_budget_caps_a_longer_estimate(self) -> None:
        loose = self.estimate.estimate(dataset(), self.form(), [0])
        capped = self.estimate.estimate(dataset(), self.form(time=0.1), [0])
        # 예산(360초)이 그냥 두었을 때보다 짧아야 상한이 의미가 있다.
        self.assertGreater(loose["total_time_s"], 360)
        self.assertEqual(capped["total_time_s"], 360.0)

    def test_zero_means_off(self) -> None:
        off = self.estimate.estimate(dataset(), self.form(time=0.0), [0])
        none = self.estimate.estimate(dataset(), self.form(), [0])
        self.assertEqual(off["total_time_s"], none["total_time_s"])

    def test_budget_does_not_stretch_a_shorter_run(self) -> None:
        """에폭을 다 돌아 예산보다 먼저 끝나면 예산은 아무것도 하지 않는다."""
        short = self.estimate.estimate(dataset(), self.form(epochs=1), [0])
        with_budget = self.estimate.estimate(dataset(), self.form(epochs=1, time=5.0), [0])
        self.assertEqual(short["total_time_s"], with_budget["total_time_s"])

    def test_capped_range_is_tight_and_starts_at_the_budget(self) -> None:
        """상한이 걸리면 남은 불확실성은 보정 오차가 아니라 마지막 마무리뿐이다."""
        capped = self.estimate.estimate(dataset(), self.form(time=0.1), [0])
        low, high = capped["range_s"]
        self.assertEqual(low, 360)
        self.assertGreater(high, low)
        # 한 에폭 이내로 본다. 보정 오차 배수(2.5배)를 그대로 쓰면 예산의 의미가 사라진다.
        self.assertLessEqual(high - low, round(capped["epoch_time_s"]) + 1)

    def test_capping_is_explained(self) -> None:
        capped = self.estimate.estimate(dataset(), self.form(time=0.1), [0])
        joined = " ".join(capped["assumptions"])
        self.assertIn("시간 예산", joined)
        # 엄격한 상한이라고 오해하게 두면 안 된다.
        self.assertIn("넘습니다", joined)


if __name__ == "__main__":
    unittest.main()
