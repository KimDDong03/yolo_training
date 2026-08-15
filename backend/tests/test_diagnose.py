"""학습 후 진단 계산 검증.

여기 숫자가 틀리면 사용자는 엉뚱한 클래스를 붙잡고 데이터를 늘린다. 매칭과 임계값 추천,
그리고 "추천하면 안 되는 상황" 판정에 집중한다.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from ._support import isolate_storage  # noqa: F401  (sys.path 설정)

from app.services import diagnose  # noqa: E402


def record(gt: list[tuple[int, list[float]]], pred: list[tuple[int, float, list[float]]]) -> dict:
    return {
        "im_file": "img.jpg",
        "ori_shape": (100, 100),
        "imgsz": (100, 100),
        "ratio_pad": ((1.0, 1.0), (0, 0)),
        "gt_cls": np.array([c for c, _ in gt], dtype=int),
        "gt_xyxy": np.array([b for _, b in gt], dtype=np.float32).reshape(-1, 4),
        "p_cls": np.array([c for c, _, _ in pred], dtype=int),
        "p_conf": np.array([s for _, s, _ in pred], dtype=np.float32),
        "p_xyxy": np.array([b for _, _, b in pred], dtype=np.float32).reshape(-1, 4),
    }


BOX = [10.0, 10.0, 30.0, 30.0]
FAR = [70.0, 70.0, 90.0, 90.0]


class IouTest(unittest.TestCase):
    def test_identical_boxes(self) -> None:
        a = np.array([BOX], dtype=np.float32)
        self.assertAlmostEqual(float(diagnose.iou_matrix(a, a)[0, 0]), 1.0, places=5)

    def test_disjoint_boxes(self) -> None:
        a = np.array([BOX], dtype=np.float32)
        b = np.array([FAR], dtype=np.float32)
        self.assertEqual(float(diagnose.iou_matrix(a, b)[0, 0]), 0.0)

    def test_half_overlap(self) -> None:
        a = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        b = np.array([[5.0, 0.0, 15.0, 10.0]], dtype=np.float32)
        # 교집합 50, 합집합 150
        self.assertAlmostEqual(float(diagnose.iou_matrix(a, b)[0, 0]), 50 / 150, places=5)

    def test_empty_inputs(self) -> None:
        empty = np.zeros((0, 4), dtype=np.float32)
        self.assertEqual(diagnose.iou_matrix(empty, np.array([BOX])).shape, (0, 1))


class MatchTest(unittest.TestCase):
    def test_exact_hit(self) -> None:
        m = diagnose.match(record([(0, BOX)], [(0, 0.9, BOX)]), conf=0.25)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 0, 0))

    def test_wrong_class_is_not_a_hit(self) -> None:
        """박스가 완벽해도 클래스가 다르면 맞춘 게 아니다."""
        m = diagnose.match(record([(0, BOX)], [(1, 0.9, BOX)]), conf=0.25)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))

    def test_poorly_placed_box_misses(self) -> None:
        m = diagnose.match(record([(0, BOX)], [(0, 0.9, FAR)]), conf=0.25)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 1))

    def test_one_ground_truth_is_claimed_once(self) -> None:
        """같은 정답을 두 예측이 나눠 가질 수 없다. 두 번째는 오검출이다."""
        m = diagnose.match(record([(0, BOX)], [(0, 0.9, BOX), (0, 0.8, BOX)]), conf=0.25)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 0))

    def test_highest_confidence_claims_first(self) -> None:
        rec = record([(0, BOX)], [(0, 0.3, BOX), (0, 0.95, BOX)])
        m = diagnose.match(rec, conf=0.25)
        # 정렬 후 첫 번째(신뢰도 0.95)가 맞춘 것이어야 한다.
        self.assertTrue(m["p_hit"][0])
        self.assertAlmostEqual(float(m["p_conf"][0]), 0.95, places=5)

    def test_low_confidence_predictions_are_dropped(self) -> None:
        m = diagnose.match(record([(0, BOX)], [(0, 0.1, BOX)]), conf=0.25)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 0, 1))

    def test_no_predictions_means_all_missed(self) -> None:
        m = diagnose.match(record([(0, BOX), (1, FAR)], []), conf=0.25)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 0, 2))

    def test_background_image_with_false_positive(self) -> None:
        m = diagnose.match(record([], [(0, 0.9, BOX)]), conf=0.25)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (0, 1, 0))


def fake_metric(f1_peak_at: float, peak_value: float, classes=(0, 1)):
    """px/f1_curve 를 가진 최소한의 Metric 대역."""
    px = np.linspace(0, 1, 1000, dtype=np.float32)
    curve = np.zeros((len(classes), 1000), dtype=np.float32)
    peak = int(f1_peak_at * 999)
    for row in range(len(classes)):
        curve[row] = np.maximum(peak_value - np.abs(px - px[peak]) * peak_value, 0.0)
    return SimpleNamespace(
        px=px, f1_curve=curve, p_curve=curve.copy(), r_curve=curve.copy(),
        ap_class_index=np.array(classes),
    )


class ConfidenceTest(unittest.TestCase):
    def test_finds_the_peak(self) -> None:
        rec = diagnose.confidence_recommendation(fake_metric(0.4, 0.8), {0: "a", 1: "b"})
        self.assertAlmostEqual(rec["conf"], 0.4, places=1)
        self.assertTrue(rec["reliable"])
        self.assertIsNone(rec["message"])

    def test_degenerate_peak_is_refused(self) -> None:
        """덜 학습된 모델은 임계값 0 에서 F1 이 최대가 된다.

        산수로는 맞지만 그대로 쓰면 한 장에 수백 개가 잡힌다. 추천 대신 사실을 말해야 한다.
        """
        rec = diagnose.confidence_recommendation(fake_metric(0.0, 0.01), {0: "a", 1: "b"})
        self.assertFalse(rec["reliable"])
        self.assertIn("오검출", rec["message"])

    def test_curve_is_downsampled(self) -> None:
        """클래스마다 1000점을 그대로 실으면 리포트가 수 MB 가 된다."""
        rec = diagnose.confidence_recommendation(fake_metric(0.4, 0.8), {0: "a", 1: "b"})
        self.assertLessEqual(len(rec["curve"]), diagnose.CURVE_POINTS + 15)

    def test_per_class_peaks_are_listed(self) -> None:
        rec = diagnose.confidence_recommendation(fake_metric(0.4, 0.8), {0: "a", 1: "b"})
        self.assertEqual([p["name"] for p in rec["per_class"]], ["a", "b"])


def summary_row(name, instances, ap):
    return {
        "Class": name, "Images": 10, "Instances": instances,
        "Box-P": 0.5, "Box-R": 0.5, "Box-F1": 0.5, "mAP50": ap + 0.1, "mAP50-95": ap,
    }


class PerClassTest(unittest.TestCase):
    def test_classes_absent_from_val_are_shown_as_unevaluated(self) -> None:
        """summary() 는 정답이 없는 클래스를 통째로 뺀다.

        그대로 두면 "표에 없으니 문제 없다" 고 읽힌다.
        """
        metrics = SimpleNamespace(summary=lambda **_: [summary_row("a", 100, 0.6)])
        table = diagnose.per_class_table(metrics, {0: "a", 1: "b"})
        self.assertEqual(len(table), 2)
        missing = [r for r in table if r["name"] == "b"][0]
        self.assertFalse(missing["evaluated"])
        self.assertEqual(missing["instances"], 0)
        self.assertIsNone(missing["ap50_95"])


class WorstClassTest(unittest.TestCase):
    def build(self, pairs, extra_names=()):
        names = {i: n for i, (n, _, _) in enumerate(pairs)}
        for j, n in enumerate(extra_names):
            names[len(pairs) + j] = n
        metrics = SimpleNamespace(
            summary=lambda **_: [summary_row(n, inst, ap) for n, inst, ap in pairs]
        )
        return diagnose.per_class_table(metrics, names)

    def test_uniformly_weak_model_reports_one_message(self) -> None:
        """전부 약하면 클래스 문제가 아니다. 다 나열하면 무엇부터 할지 알 수 없다."""
        table = self.build([("a", 100, 0.05), ("b", 100, 0.06), ("c", 100, 0.04)])
        worst = diagnose.worst_classes(table)
        self.assertEqual(len(worst), 1)
        self.assertIsNone(worst[0]["name"])
        self.assertIn("모든 클래스가 약합니다", worst[0]["message"])

    def test_one_weak_class_among_good_ones_is_named(self) -> None:
        table = self.build([("good", 500, 0.8), ("fine", 500, 0.75), ("weak", 20, 0.1)])
        worst = diagnose.worst_classes(table)
        self.assertEqual([w["name"] for w in worst], ["weak"])
        self.assertIn("데이터를 늘리는", worst[0]["message"])

    def test_plenty_of_data_but_weak_suggests_labels(self) -> None:
        table = self.build([("good", 500, 0.8), ("fine", 500, 0.75), ("weak", 900, 0.1)])
        worst = diagnose.worst_classes(table)
        self.assertIn("라벨 기준", worst[0]["message"])

    def test_worst_list_is_capped(self) -> None:
        pairs = [("good", 500, 0.9)] + [(f"w{i}", 50, 0.05) for i in range(6)]
        worst = diagnose.worst_classes(self.build(pairs))
        self.assertLessEqual(len(worst), diagnose.WORST_CAP)

    def test_unevaluated_classes_are_always_reported(self) -> None:
        table = self.build([("a", 100, 0.8), ("b", 100, 0.7)], extra_names=("ghost",))
        worst = diagnose.worst_classes(table)
        self.assertIn("ghost", [w["name"] for w in worst])
        self.assertIn("검증 셋", [w for w in worst if w["name"] == "ghost"][0]["message"])


class GalleryTest(unittest.TestCase):
    def test_ranked_by_mistakes_with_misses_weighted_higher(self) -> None:
        two_missed = record([(0, BOX), (0, FAR)], [])
        two_missed["im_file"] = "missed.jpg"
        one_false = record([], [(0, 0.9, BOX)])
        one_false["im_file"] = "false.jpg"
        gallery, total = diagnose.build_gallery([one_false, two_missed], 0.25, {0: "a"})
        self.assertEqual([g["name"] for g in gallery], ["missed.jpg", "false.jpg"])
        self.assertEqual(total, 2)

    def test_perfect_images_are_left_out(self) -> None:
        perfect = record([(0, BOX)], [(0, 0.9, BOX)])
        gallery, total = diagnose.build_gallery([perfect], 0.25, {0: "a"})
        self.assertEqual(gallery, [])
        self.assertEqual(total, 0)

    def test_box_states_are_labelled(self) -> None:
        rec = record([(0, BOX), (0, FAR)], [(0, 0.9, BOX), (1, 0.8, [40.0, 40.0, 50.0, 50.0])])
        gallery, _ = diagnose.build_gallery([rec], 0.25, {0: "a", 1: "b"})
        states = {b["state"] for b in gallery[0]["gt"]}
        self.assertEqual(states, {"hit", "miss"})
        self.assertIn("false", {b["state"] for b in gallery[0]["pred"]})

    def test_gallery_is_capped(self) -> None:
        records = []
        for i in range(diagnose.GALLERY_CAP + 20):
            r = record([(0, BOX)], [])
            r["im_file"] = f"img{i}.jpg"
            records.append(r)
        gallery, total = diagnose.build_gallery(records, 0.25, {0: "a"})
        self.assertEqual(len(gallery), diagnose.GALLERY_CAP)
        self.assertEqual(total, diagnose.GALLERY_CAP + 20)

    def test_boxes_per_image_are_capped(self) -> None:
        """conf 0.001 로 검증하면 한 장에 수백 개가 나온다. 다 그리면 사진이 안 보인다."""
        many = [(0, 0.9, [float(i), float(i), float(i) + 5, float(i) + 5])
                for i in range(diagnose.BOXES_PER_IMAGE + 30)]
        gallery, _ = diagnose.build_gallery([record([(0, FAR)], many)], 0.25, {0: "a"})
        self.assertLessEqual(len(gallery[0]["pred"]), diagnose.BOXES_PER_IMAGE)

    def test_display_boxes_are_normalised(self) -> None:
        gallery, _ = diagnose.build_gallery([record([(0, BOX)], [])], 0.25, {0: "a"})
        for value in gallery[0]["gt"][0]["box"]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
