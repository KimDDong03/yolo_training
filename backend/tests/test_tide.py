"""오류 분해 검증.

여기 숫자가 틀리면 사용자는 엉뚱한 것을 고친다 — 라벨을 손봐야 할 때 해상도를 올리거나,
데이터를 늘려야 할 때 NMS 를 만진다. 그래서 기대값을 손으로 계산하지 않고 실제
ap_per_class 로 대조해 못 박아 둔다.

픽스처는 셋 다 목적이 다르다. A 는 여섯 유형을 한 번에 덮고, B 는 놓침을 고칠 때 분모가
바뀌는 함정을, C 는 "고쳐도 중복이 되는 검출" 을 지우면 안 된다는 것을, D 는 판정 순서를
지킨다.
"""

from __future__ import annotations

import unittest

import numpy as np

from ._support import isolate_storage  # noqa: F401  (sys.path 설정)

from app.services import tide  # noqa: E402


def record(
    name: str,
    gt: list[tuple[int, list[float]]],
    pred: list[tuple[int, float, list[float]]],
) -> dict:
    return {
        "im_file": name,
        "ori_shape": (100, 100),
        "imgsz": (100, 100),
        "ratio_pad": ((1.0, 1.0), (0, 0)),
        "gt_cls": np.array([c for c, _ in gt], dtype=int),
        "gt_xyxy": np.array([b for _, b in gt], dtype=np.float32).reshape(-1, 4),
        "p_cls": np.array([c for c, _, _ in pred], dtype=int),
        "p_conf": np.array([s for _, s, _ in pred], dtype=np.float32),
        "p_xyxy": np.array([b for _, _, b in pred], dtype=np.float32).reshape(-1, 4),
    }


A = [10.0, 10.0, 30.0, 30.0]
B = [50.0, 10.0, 70.0, 30.0]
C = [50.0, 50.0, 90.0, 90.0]
AS = [20.0, 10.0, 40.0, 30.0]  # A 와 IoU 1/3
CS = [70.0, 50.0, 110.0, 90.0]  # C 와 IoU 1/3
D = [0.0, 80.0, 10.0, 90.0]  # 무엇과도 안 겹친다


def fixture_a() -> list[dict]:
    """이미지 3, 정답 5, 예측 8. 여섯 유형이 한 번씩 나오고 소수 클래스가 사라진다."""
    return [
        record("1.jpg", [(0, A), (0, B)], [(0, 0.90, A), (0, 0.85, A), (0, 0.80, B)]),
        record(
            "2.jpg",
            [(1, A), (1, C)],
            [(1, 0.75, AS), (0, 0.95, C), (2, 0.65, CS), (2, 0.55, C)],
        ),
        record("3.jpg", [(2, A)], [(0, 0.88, D)]),
    ]


NAMES = {0: "a", 1: "b", 2: "c"}


class ClassifyTest(unittest.TestCase):
    def test_every_error_type_is_labelled(self) -> None:
        """딱지는 이미지마다 신뢰도 내림차순으로 붙는다."""
        dets, _ = tide.classify(fixture_a())
        kinds = [
            None if code == tide._NO_ERROR else [k for k, v in tide._CODES.items() if v == code][0]
            for code in dets["err"]
        ]
        self.assertEqual(
            kinds,
            [None, "dupe", None, "cls", "loc", "both", "cls", "bkg"],
        )

    def test_evidence_points_at_the_right_ground_truth(self) -> None:
        dets, _ = tide.classify(fixture_a())
        # 전역 정답 index: 0,1 = 1.jpg / 2,3 = 2.jpg / 4 = 3.jpg
        self.assertEqual(int(dets["gt"][1]), 0)  # 중복은 이미 잡힌 첫 정답을 또 잡은 것
        self.assertEqual(int(dets["gt"][3]), 3)  # 클래스 오류가 노리는 정답
        self.assertEqual(int(dets["gt"][4]), 2)  # 위치 오류가 노리는 정답
        self.assertEqual(int(dets["gt"][6]), 3)  # 두 번째 클래스 오류가 같은 정답을 노린다

    def test_only_unexplained_ground_truth_counts_as_missed(self) -> None:
        """위치·클래스 오류가 덮은 정답은 놓친 게 아니다 — 고치면 잡힌다."""
        _, gts = tide.classify(fixture_a())
        self.assertEqual(list(np.flatnonzero(gts["miss"])), [4])

    def test_duplicate_always_points_at_a_claimed_ground_truth(self) -> None:
        dets, gts = tide.classify(fixture_a())
        for row in np.flatnonzero(dets["err"] == tide._CODES["dupe"]):
            self.assertTrue(gts["taken"][dets["gt"][row]])

    def test_class_and_box_errors_always_point_at_a_free_ground_truth(self) -> None:
        """이 불변식이 깨지면 fix 의 claim 절차가 의미를 잃는다."""
        dets, gts = tide.classify(fixture_a())
        for kind in ("cls", "loc"):
            for row in np.flatnonzero(dets["err"] == tide._CODES[kind]):
                self.assertFalse(gts["taken"][dets["gt"][row]])

    def test_best_iou_counts_matched_predictions_too(self) -> None:
        """놓침 판정은 못 맞춘 예측만 보지만, 라벨 오류 후보는 전부를 봐야 한다."""
        _, gts = tide.classify(fixture_a())
        self.assertAlmostEqual(float(gts["best_iou"][0]), 1.0, places=5)
        self.assertEqual(float(gts["best_iou"][4]), 0.0)


class ClassificationOrderTest(unittest.TestCase):
    """판정 순서 회귀 방지. 둘 다 순서를 되돌리면 실패한다."""

    def test_class_error_beats_duplicate(self) -> None:
        """이미 잡힌 같은 종류와 0.6, 아직 안 잡힌 다른 종류와 0.9 — 중복이 아니라 클래스 오류다."""
        taken_box = [0.0, 0.0, 100.0, 60.0]  # 아래 예측과 IoU 0.6
        other_box = [0.0, 0.0, 100.0, 90.0]  # 아래 예측과 IoU 0.9
        wide = [0.0, 0.0, 100.0, 100.0]
        dets, _ = tide.classify(
            [record("x.jpg", [(0, taken_box), (1, other_box)],
                    [(0, 0.95, taken_box), (0, 0.90, wide)])]
        )
        self.assertEqual(int(dets["err"][1]), tide._CODES["cls"])

    def test_overlap_with_a_claimed_box_is_not_a_localisation_error(self) -> None:
        """미점유 정답이 없으면 위치 오류가 될 수 없다. 나머지는 전부 both 다."""
        taken_box = [0.0, 0.0, 100.0, 30.0]  # 아래 예측과 IoU 0.3
        wide = [0.0, 0.0, 100.0, 100.0]
        dets, _ = tide.classify(
            [record("x.jpg", [(0, taken_box)], [(0, 0.95, taken_box), (0, 0.90, wide)])]
        )
        self.assertEqual(int(dets["err"][1]), tide._CODES["both"])


def deltas(report: dict) -> dict[str, float]:
    return {e["kind"]: e["dap"] for e in report["errors"]}


class BreakdownTest(unittest.TestCase):
    def test_fixture_a_matches_hand_checked_numbers(self) -> None:
        report = tide.error_breakdown(fixture_a(), NAMES)
        self.assertAlmostEqual(report["baseline_map50"], 0.149167, places=5)
        self.assertEqual(report["baseline_classes"], [0, 1, 2])
        self.assertEqual(
            deltas(report),
            # 손계산 값 0.264167 / 0.016667 을 리포트 정밀도(소수 5자리)로 자른 것.
            {
                "cls": 0.26417,
                "loc": 0.165,
                "both": 0.0,
                "dupe": 0.01667,
                "bkg": 0.01667,
                "miss": 0.0,
            },
        )

    def test_counts_are_reported_next_to_the_deltas(self) -> None:
        """상승분이 0 이어도 건수는 남아야 한다 — 사용자가 보는 건 대개 건수다."""
        counts = {e["kind"]: e["count"] for e in tide.error_breakdown(fixture_a(), NAMES)["errors"]}
        self.assertEqual(counts, {"cls": 2, "loc": 1, "both": 1, "dupe": 1, "bkg": 1, "miss": 1})

    def test_vanished_class_is_dropped_from_both_averages(self) -> None:
        """놓침을 고치면 정답이 하나도 안 남는 클래스가 생긴다. 분모가 줄면 상승분이 부풀려진다."""
        report = tide.error_breakdown(
            [record("1.jpg", [(0, A), (0, B), (1, C)], [(0, 0.90, A)])], NAMES
        )
        miss = [e for e in report["errors"] if e["kind"] == "miss"][0]
        self.assertAlmostEqual(miss["dap"], 0.5, places=5)
        self.assertAlmostEqual(miss["dap_naive"], 0.7475, places=5)
        self.assertEqual(miss["dropped_classes"], [1])

    def test_unfixable_duplicate_still_costs_precision(self) -> None:
        """고쳐 봐야 중복이 되는 검출을 지우면 상승분이 0.333 이 아니라 0.5 로 부푼다."""
        report = tide.error_breakdown(
            [record("1.jpg", [(1, A), (1, B)],
                    [(0, 0.90, A), (0, 0.80, A), (1, 0.70, B)])],
            NAMES,
        )
        self.assertAlmostEqual(report["baseline_map50"], 0.495, places=5)
        self.assertAlmostEqual(deltas(report)["cls"], 0.333333, places=5)

    def test_shares_add_up_over_the_positive_deltas(self) -> None:
        report = tide.error_breakdown(fixture_a(), NAMES)
        shares = [e["share"] for e in report["errors"] if e["share"]]
        self.assertAlmostEqual(sum(shares), 1.0, places=3)

    def test_confusion_pairs_name_both_sides(self) -> None:
        report = tide.error_breakdown(fixture_a(), NAMES)
        self.assertEqual(report["confusion_pairs"][0]["gt"], "b")
        self.assertIn(report["confusion_pairs"][0]["pred"], {"a", "c"})

    def test_low_confidence_class_errors_are_left_out_of_confusion_pairs(self) -> None:
        """신뢰도 0.001 로 검증하므로 다 세면 배포에서 보지도 않을 오류가 목록을 채운다."""
        report = tide.error_breakdown(
            [record("1.jpg", [(1, A)], [(0, 0.01, A)])], NAMES
        )
        self.assertEqual(report["confusion_pairs"], [])

    def test_per_class_counts_split_predictions_and_ground_truth(self) -> None:
        rows = {r["name"]: r for r in tide.error_breakdown(fixture_a(), NAMES)["per_class_counts"]}
        self.assertEqual(rows["a"]["counts"]["cls"], 1)  # 모델이 a 라고 부른 클래스 오류
        self.assertEqual(rows["c"]["counts"]["miss"], 1)  # 놓친 것은 정답 클래스로 센다
        # 클래스 번호가 "cls" 오류 건수에 덮이면 안 된다.
        self.assertEqual(rows["c"]["cls"], 2)

    def test_calls_ap_per_class_exactly_seven_times(self) -> None:
        """각 fix 는 배열 편집이지 재매칭이 아니다. 이 숫자가 늘면 그 설계가 깨진 것이다."""
        from ultralytics.utils import metrics

        original = metrics.ap_per_class
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        metrics.ap_per_class = counting
        try:
            tide.error_breakdown(fixture_a(), NAMES)
        finally:
            metrics.ap_per_class = original
        self.assertEqual(len(calls), 7)


class EmptyInputTest(unittest.TestCase):
    def test_no_images_at_all(self) -> None:
        report = tide.error_breakdown([], NAMES)
        self.assertEqual(report["baseline_map50"], 0.0)
        self.assertEqual(report["baseline_classes"], [])
        self.assertTrue(all(e["dap"] == 0.0 for e in report["errors"]))

    def test_images_without_ground_truth(self) -> None:
        """정답이 없는 사진의 검출은 전부 배경 오검출이다."""
        report = tide.error_breakdown([record("1.jpg", [], [(0, 0.9, A)])], NAMES)
        counts = {e["kind"]: e["count"] for e in report["errors"]}
        self.assertEqual(counts["bkg"], 1)
        self.assertEqual(report["baseline_map50"], 0.0)

    def test_images_without_predictions(self) -> None:
        report = tide.error_breakdown([record("1.jpg", [(0, A)], [])], NAMES)
        counts = {e["kind"]: e["count"] for e in report["errors"]}
        self.assertEqual(counts["miss"], 1)

    def test_every_error_carries_a_korean_sentence(self) -> None:
        for error in tide.error_breakdown(fixture_a(), NAMES)["errors"]:
            self.assertTrue(error["advice"].strip())
            self.assertFalse(error["advice"].isascii())


if __name__ == "__main__":
    unittest.main()
