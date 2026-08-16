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


class RetiredPhantomLabelTest(unittest.TestCase):
    """접은 신호가 조용히 되살아나지 않게 막는다.

    "정답 자리에 conf 0.001 까지 낮춰도 검출이 없으니 라벨이 이상하다" 는 판정이었다.
    데이터셋 7종 40건 전수 판정에서 이 레포의 3종은 0/19 였다. 근거는 .codex/phase-5.md.
    """

    def test_a_ground_truth_with_no_detection_produces_nothing(self) -> None:
        report = issues([record("1.jpg", [(0, BIG)], [(0, 0.9, FAR)])])
        self.assertNotIn("phantom_label", kinds(report))
        # 그 사진에서 나오는 후보는 배경 오검출 쪽뿐이어야 한다.
        self.assertNotIn("phantom_label", [k["kind"] for k in report["kinds"]])

    def test_the_kind_is_gone_from_every_table(self) -> None:
        self.assertNotIn("phantom_label", label_issues.LABELS)
        self.assertNotIn("phantom_label", label_issues.KIND_ORDER)
        self.assertNotIn("phantom_label", label_issues.MODEL_KINDS)

    def test_its_private_thresholds_are_gone_too(self) -> None:
        """phantom 전용이던 상수. 남겨 두면 다음 사람이 규칙이 있는 줄 안다."""
        for dead in ("NO_SIGNAL_IOU", "MIN_CLASS_RECALL"):
            self.assertFalse(hasattr(label_issues, dead), dead)


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
        # 예측 하나를 겹쳐 둬야 중복만 세진다.
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

    def test_scope_note_names_the_failure_mode_we_measured(self) -> None:
        # 실측 오탐 9건은 전부 학습 클래스에 없는 물체였다. 사용자가 화면에서 그 패턴을
        # 알아보지 못하면 목록 전체를 믿거나 전체를 버리는 둘 중 하나가 된다.
        self.assertIn("학습 클래스 목록에 없는 물체", issues([])["scope_note"])

    def test_kind_order_follows_measured_precision(self) -> None:
        # 실측(후보 47건 전수): wrong_class 7/7, unlabeled_object 2/2, missing_label 26/36.
        # 순서를 되돌리려면 .codex/phase-5.md 의 표를 먼저 읽어야 한다.
        order = label_issues.KIND_ORDER
        self.assertLess(order.index("wrong_class"), order.index("missing_label"))
        self.assertEqual(set(order), set(label_issues.LABELS))

    def test_retired_kinds_are_stripped_from_stored_reports(self) -> None:
        """저장소에 이미 있는 리포트를 사용자는 재분석 없이 열어 본다.

        접은 종류를 그대로 두면 접기로 한 바로 그 후보가 계속 뜬다.
        건수도 같이 줄여야 요약 줄과 next_actions 가 같은 숫자를 말한다.
        """
        from app.api import runs

        stored = {
            "total": 5, "shown": 3,
            "kinds": [
                {"kind": "missing_label", "label": "라벨 누락 의심", "count": 3},
                {"kind": "phantom_label", "label": "빈 자리 정답", "count": 2},
            ],
            "items": [
                {"name": "1.jpg", "findings": [
                    {"kind": "missing_label", "message": "살아 있는 문장"},
                    {"kind": "phantom_label", "message": "빈 자리 정답 …"},
                ]},
                # phantom 만 있던 사진은 통째로 빠져야 한다.
                {"name": "2.jpg", "findings": [{"kind": "phantom_label", "message": "…"}]},
            ],
        }
        runs._drop_retired_kinds(stored)
        self.assertEqual([k["kind"] for k in stored["kinds"]], ["missing_label"])
        self.assertEqual(stored["total"], 3)
        self.assertEqual([i["name"] for i in stored["items"]], ["1.jpg"])
        self.assertEqual(
            [f["kind"] for f in stored["items"][0]["findings"]], ["missing_label"]
        )
        self.assertEqual(stored["shown"], 1)

    def test_live_kinds_are_left_exactly_as_stored(self) -> None:
        """다른 종류의 설명에는 그 리포트에서만 나오는 건수·신뢰도가 박혀 있다."""
        from app.api import runs

        original = "이 사진에는 a 정답이 3개 있고 모델이 그중 2개를 맞췄습니다."
        stored = {
            "total": 1, "shown": 1,
            "kinds": [{"kind": "missing_label", "label": "라벨 누락 의심", "count": 1}],
            "items": [{"findings": [{"kind": "missing_label", "message": original}]}],
        }
        runs._drop_retired_kinds(stored)
        self.assertEqual(stored["items"][0]["findings"][0]["message"], original)
        self.assertEqual(stored["total"], 1)
        self.assertEqual(stored["shown"], 1)

    def test_every_finding_carries_a_korean_sentence(self) -> None:
        report = issues(
            [
                record("1.jpg", [(0, BIG)], [(0, 0.95, BIG), (0, 0.95, FAR)]),
                record("2.jpg", [(1, BIG)], [(0, 0.95, BIG)]),
                record("3.jpg", [(0, BIG), (1, BIG)], []),
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
