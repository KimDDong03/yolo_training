"""라벨 오류 후보 판정 검증.

여기서 오탐이 나면 사용자는 멀쩡한 라벨을 고치거나, 몇 번 헛걸음한 뒤 목록 전체를 무시한다.
그래서 "잡히는가" 보다 "안 잡혀야 할 것이 안 잡히는가" 를 더 많이 확인한다.
"""

from __future__ import annotations

import unittest

import numpy as np

from ._support import isolate_storage  # noqa: F401  (sys.path 설정)

from app.services import label_issues, tide  # noqa: E402


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


# 넓이가 이미지의 0.2% 를 넘어야 후보가 된다. 20x20 = 4% 라 넉넉하다.
BIG = [10.0, 10.0, 30.0, 30.0]
FAR = [60.0, 60.0, 80.0, 80.0]
TINY = [0.0, 0.0, 3.0, 3.0]  # 0.09% — 작아서 후보에서 빠져야 한다
NAMES = {0: "a", 1: "b"}


def strong(cls: int, name: str = "a") -> dict:
    """모델 판단을 근거로 써도 되는 클래스 행."""
    return {
        "cls": cls, "name": name, "images": 50, "instances": 100,
        "precision": 0.9, "recall": 0.9, "f1": 0.9, "ap50": 0.9,
        "ap50_95": 0.8, "evaluated": True,
    }


def issues(records, per_class=None, map50=0.8, reliable=True):
    dets, gts = tide.classify(records)
    return label_issues.build(
        records,
        dets,
        gts,
        NAMES,
        per_class or [strong(0, "a"), strong(1, "b")],
        {"map50": map50},
        conf_reliable=reliable,
    )


def kinds(report) -> list[str]:
    return [f["kind"] for item in report["items"] for f in item["findings"]]


class MissingLabelTest(unittest.TestCase):
    def test_confident_detection_beside_a_correct_one_is_flagged(self) -> None:
        """이 사진에서 이 클래스를 모델이 실제로 맞혔다면, 추가 검출은 라벨 누락일 수 있다."""
        report = issues([record("1.jpg", [(0, BIG)], [(0, 0.95, BIG), (0, 0.90, FAR)])])
        self.assertIn("missing_label", kinds(report))

    def test_detection_below_the_confidence_line_is_left_alone(self) -> None:
        report = issues([record("1.jpg", [(0, BIG)], [(0, 0.95, BIG), (0, 0.69, FAR)])])
        self.assertNotIn("missing_label", kinds(report))

    def test_without_a_correct_detection_nearby_a_much_higher_bar_applies(self) -> None:
        """뒷받침할 근거가 없으면 신뢰도만이 유일한 근거라 기준이 올라간다."""
        weak = issues([record("1.jpg", [(1, BIG)], [(0, 0.80, FAR)])])
        self.assertNotIn("unlabeled_object", kinds(weak))
        sure = issues([record("1.jpg", [(1, BIG)], [(0, 0.95, FAR)])])
        self.assertIn("unlabeled_object", kinds(sure))

    def test_class_the_model_over_predicts_is_never_used_as_evidence(self) -> None:
        sloppy = dict(strong(0, "a"), precision=0.2)
        report = issues(
            [record("1.jpg", [(0, BIG)], [(0, 0.95, BIG), (0, 0.95, FAR)])],
            per_class=[sloppy, strong(1, "b")],
        )
        self.assertNotIn("missing_label", kinds(report))

    def test_class_with_too_few_instances_is_never_used_as_evidence(self) -> None:
        rare = dict(strong(0, "a"), instances=5)
        report = issues(
            [record("1.jpg", [(0, BIG)], [(0, 0.95, BIG), (0, 0.95, FAR)])],
            per_class=[rare, strong(1, "b")],
        )
        self.assertNotIn("missing_label", kinds(report))

    def test_tiny_boxes_are_never_flagged(self) -> None:
        """작은 물체는 모델이 원래 못 찾는다. 그걸 라벨 오류라고 부르면 안 된다."""
        report = issues([record("1.jpg", [(0, BIG)], [(0, 0.95, BIG), (0, 0.95, TINY)])])
        self.assertNotIn("missing_label", kinds(report))


class WrongClassTest(unittest.TestCase):
    def test_tight_overlap_with_another_class_is_flagged(self) -> None:
        report = issues([record("1.jpg", [(1, BIG)], [(0, 0.95, BIG)])])
        self.assertIn("wrong_class", kinds(report))
        finding = [f for i in report["items"] for f in i["findings"]][0]
        self.assertEqual(finding["ref_name"], "b")
        self.assertIsNotNone(finding["ref_box"])

    def test_a_loose_box_is_not_a_class_mix_up(self) -> None:
        """IoU 0.5~0.75 는 '옆 물체를 잘못 집은 것' 과 구분되지 않는다."""
        loose = [10.0, 10.0, 30.0, 26.0]  # BIG 과 IoU 0.8 미만
        report = issues([record("1.jpg", [(1, BIG)], [(0, 0.95, loose)])])
        overlap = [f["iou"] for i in report["items"] for f in i["findings"]]
        self.assertTrue(all(v >= label_issues.TIGHT_IOU for v in overlap))

    def test_low_confidence_class_errors_are_left_alone(self) -> None:
        report = issues([record("1.jpg", [(1, BIG)], [(0, 0.5, BIG)])])
        self.assertNotIn("wrong_class", kinds(report))


class PhantomLabelTest(unittest.TestCase):
    def test_ground_truth_nothing_overlapped_is_flagged(self) -> None:
        """신뢰도 0.001 까지 낮춰도 아무것도 안 걸리면 그 자리엔 물체 비슷한 것도 없다."""
        report = issues([record("1.jpg", [(0, BIG)], [(0, 0.9, FAR)])])
        self.assertIn("phantom_label", kinds(report))

    def test_ground_truth_something_overlapped_is_not_flagged(self) -> None:
        near = [12.0, 12.0, 30.0, 30.0]
        report = issues([record("1.jpg", [(0, BIG)], [(1, 0.4, near)])])
        self.assertNotIn("phantom_label", kinds(report))

    def test_needs_a_class_the_model_normally_finds(self) -> None:
        blind = dict(strong(0, "a"), recall=0.2)
        report = issues(
            [record("1.jpg", [(0, BIG)], [(0, 0.9, FAR)])],
            per_class=[blind, strong(1, "b")],
        )
        self.assertNotIn("phantom_label", kinds(report))

    def test_tiny_ground_truth_is_never_flagged(self) -> None:
        report = issues([record("1.jpg", [(0, TINY)], [(0, 0.9, FAR)])])
        self.assertNotIn("phantom_label", kinds(report))


class LabelOnlyTest(unittest.TestCase):
    def test_overlapping_same_class_boxes_are_a_duplicate(self) -> None:
        twin = [10.0, 10.0, 29.0, 30.0]  # BIG 과 IoU 0.95
        report = issues([record("1.jpg", [(0, BIG), (0, twin)], [])])
        self.assertIn("duplicate_gt", kinds(report))

    def test_moderately_overlapping_boxes_are_left_alone(self) -> None:
        """서로 다른 물체가 0.8 이나 겹치는 일은 드물다. 낮추면 군중 사진에서 오탐이 난다."""
        apart = [10.0, 10.0, 30.0, 25.0]  # BIG 과 IoU 0.75
        report = issues([record("1.jpg", [(0, BIG), (0, apart)], [])])
        self.assertNotIn("duplicate_gt", kinds(report))

    def test_two_classes_on_one_object_are_flagged(self) -> None:
        report = issues([record("1.jpg", [(0, BIG), (1, BIG)], [])])
        self.assertIn("conflicting_gt", kinds(report))


class GuardTest(unittest.TestCase):
    def test_untrustworthy_model_yields_only_label_only_signals(self) -> None:
        report = issues(
            [record("1.jpg", [(0, BIG), (1, BIG)], [(0, 0.95, FAR)])], reliable=False
        )
        self.assertFalse(report["model_evidence"])
        self.assertIn("학습", report["reason"])
        self.assertEqual(set(kinds(report)), {"conflicting_gt"})

    def test_low_map_model_yields_only_label_only_signals(self) -> None:
        report = issues([record("1.jpg", [(1, BIG)], [(0, 0.95, BIG)])], map50=0.2)
        self.assertFalse(report["model_evidence"])
        self.assertIn("mAP50", report["reason"])
        self.assertEqual(kinds(report), [])

    def test_healthy_model_keeps_the_reason_empty(self) -> None:
        report = issues([record("1.jpg", [(1, BIG)], [(0, 0.95, BIG)])])
        self.assertTrue(report["model_evidence"])
        self.assertIsNone(report["reason"])


class ShapeTest(unittest.TestCase):
    def build_many(self, count: int) -> list[dict]:
        # 예측 하나를 겹쳐 둬야 '빈 자리 정답' 이 섞이지 않아 중복만 세진다.
        twin = [10.0, 10.0, 29.0, 30.0]
        return [
            record(f"{i}.jpg", [(0, BIG), (0, twin)], [(0, 0.95, BIG)])
            for i in range(count)
        ]

    def test_one_kind_cannot_take_over_the_list(self) -> None:
        report = issues(self.build_many(label_issues.ISSUE_KIND_CAP + 15))
        self.assertEqual(len(kinds(report)), label_issues.ISSUE_KIND_CAP)

    def test_total_counts_everything_before_the_caps(self) -> None:
        report = issues(self.build_many(40))
        self.assertEqual(report["total"], 40)
        self.assertLessEqual(len(report["items"]), label_issues.ISSUE_IMAGE_CAP)
        self.assertLess(report["shown"], report["total"])

    def test_kind_summary_lists_only_what_was_found(self) -> None:
        report = issues([record("1.jpg", [(0, BIG), (1, BIG)], [(0, 0.95, BIG)])])
        self.assertEqual([k["kind"] for k in report["kinds"]], ["conflicting_gt"])
        self.assertEqual(report["kinds"][0]["label"], "한 물체에 두 클래스")

    def test_boxes_are_normalised_and_context_is_carried(self) -> None:
        report = issues([record("1.jpg", [(1, BIG)], [(0, 0.95, BIG)])])
        item = report["items"][0]
        for value in item["findings"][0]["box"]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        # 오버레이는 갤러리와 같은 규약이라 프론트 코드를 그대로 쓴다.
        self.assertEqual({b["state"] for b in item["gt"]}, {"miss"})
        self.assertEqual({b["state"] for b in item["pred"]}, {"false"})

    def test_scope_note_says_it_only_looked_at_validation(self) -> None:
        note = issues([])["scope_note"]
        self.assertIn("val", note)
        self.assertIn("학습", note)

    def test_every_finding_carries_a_korean_sentence(self) -> None:
        report = issues(
            [
                record("1.jpg", [(0, BIG)], [(0, 0.95, BIG), (0, 0.95, FAR)]),
                record("2.jpg", [(1, BIG)], [(0, 0.95, BIG)]),
                record("3.jpg", [(0, BIG), (1, BIG)], []),
                record("4.jpg", [(0, BIG)], [(0, 0.9, FAR)]),
            ]
        )
        found = [f for i in report["items"] for f in i["findings"]]
        self.assertTrue(found)
        for finding in found:
            self.assertTrue(finding["message"].strip())
            self.assertFalse(finding["message"].isascii())

    def test_empty_input_is_not_an_error(self) -> None:
        report = issues([])
        self.assertTrue(report["available"])
        self.assertEqual(report["total"], 0)
        self.assertEqual(report["items"], [])


if __name__ == "__main__":
    unittest.main()
