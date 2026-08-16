"""다음 행동 제안 검증.

이 문장이 사용자가 실제로 읽고 따라 하는 마지막 화면이다. 틀린 처방은 틀린 지표보다
비싸다 — 며칠을 엉뚱한 데 쓰게 만든다. 우선순위와 자리표시자 누락을 집중적으로 본다.
"""

from __future__ import annotations

import unittest

from ._support import isolate_storage  # noqa: F401  (sys.path 설정)

from app.services import next_actions  # noqa: E402


def report(
    *,
    # None 을 넘기는 테스트가 있다 — "아직 못 잰 값" 이 리포트에 그대로 들어오는 경우다.
    # 기본값만 보고 float 로 추론되면 그 테스트가 타입 오류로 잡힌다.
    map50: float | None = 0.7,
    map50_95: float | None = 0.5,
    precision: float | None = 0.7,
    recall: float | None = 0.6,
    instances: int = 1000,
    reliable: bool = True,
    conf: float | None = 0.25,
    tide: dict | None = None,
    labels: dict | None = None,
    worst: tuple = (),
) -> dict:
    return {
        "overall": {
            "images": 100, "instances": instances, "precision": precision,
            "recall": recall, "map50": map50, "map50_95": map50_95,
        },
        "conf_recommendation": {"reliable": reliable, "conf": conf},
        "tide": tide,
        "label_issues": labels,
        "worst_classes": list(worst),
    }


def breakdown(dominant: str, dap: float = 0.20, rest: float = 0.01, **counts) -> dict:
    kinds = ("cls", "loc", "both", "dupe", "bkg", "miss")
    return {
        "errors": [
            {
                "kind": k,
                "dap": dap if k == dominant else rest,
                "count": counts.get(k, 5),
                "count_at_conf": counts.get(f"{k}_seen", counts.get(k, 5)),
            }
            for k in kinds
        ],
        "confusion_pairs": counts.get("pairs", []),
    }


def codes(actions) -> list[str]:
    return [a["code"] for a in actions]


class ShortCircuitTest(unittest.TestCase):
    def test_untrained_model_gets_one_answer_and_nothing_else(self) -> None:
        """덜 학습된 모델의 오류 분해는 노이즈다. 다른 처방을 얹으면 오해를 부른다."""
        actions = next_actions.build(report(reliable=False, tide=breakdown("miss")))
        self.assertEqual(codes(actions), ["threshold_unusable"])

    def test_a_decent_score_with_no_usable_threshold_says_so(self) -> None:
        """mAP 는 볼만한데 임계값이 없는 상태다. mAP 가 낮다고 말하면 거짓말이 된다."""
        action = next_actions.build(report(map50=0.7, reliable=False))[0]
        self.assertEqual(action["code"], "threshold_unusable")
        self.assertIn("0.700", action["cause"])
        self.assertIn("임계값", action["cause"])

    def test_very_low_map_also_short_circuits(self) -> None:
        actions = next_actions.build(report(map50=0.05, tide=breakdown("miss")))
        self.assertEqual(codes(actions), ["model_not_ready"])

    def test_missing_map_is_treated_as_not_ready(self) -> None:
        self.assertEqual(codes(next_actions.build(report(map50=None))), ["model_not_ready"])


class DominantErrorTest(unittest.TestCase):
    def first(self, dominant: str, **kw):
        return next_actions.build(report(tide=breakdown(dominant, **kw)))[0]

    def test_missed_objects_point_at_recall(self) -> None:
        action = self.first("miss")
        self.assertEqual(action["code"], "miss_dominant")
        self.assertIn("imgsz", action["fix"])
        self.assertIn("신뢰도", action["fix"])

    def test_background_detections_point_at_precision(self) -> None:
        action = self.first("bkg")
        self.assertEqual(action["code"], "bkg_dominant")
        self.assertIn("올리면", action["fix"])

    def test_localisation_points_at_resolution_and_augmentation(self) -> None:
        action = self.first("loc")
        self.assertEqual(action["code"], "loc_dominant")
        self.assertIn("imgsz", action["fix"])
        self.assertIn("close_mosaic", action["fix"])

    def test_class_confusion_names_the_worst_pair(self) -> None:
        action = self.first("cls", pairs=[{"pred": "긁힘", "gt": "찍힘", "count": 12}])
        self.assertEqual(action["code"], "cls_dominant")
        self.assertIn("긁힘 ↔ 찍힘", action["cause"])
        self.assertIn("12", action["cause"])

    def test_class_confusion_without_pairs_uses_the_plain_wording(self) -> None:
        """예전 리포트에는 혼동 쌍이 없다. 문장에 '-' 가 새면 안 된다."""
        action = self.first("cls")
        self.assertEqual(action["code"], "cls_dominant_plain")
        self.assertNotIn("-", action["cause"])

    def test_both_axes_wrong_points_at_more_training(self) -> None:
        action = self.first("both")
        self.assertEqual(action["code"], "both_dominant")
        self.assertIn("에폭", action["fix"])

    def test_a_thin_lead_is_not_dominant(self) -> None:
        """비슷비슷하면 무엇을 먼저 할지 말할 수 없다. 억지로 말하면 틀린다."""
        even = breakdown("miss", dap=0.05, rest=0.045)
        self.assertNotIn("miss_dominant", codes(next_actions.build(report(tide=even))))

    def test_a_dominant_share_of_almost_nothing_is_not_worth_a_prescription(self) -> None:
        """mAP50 0.95 짜리 모델에도 무언가는 1등이다. 비중만 보면 '학습이 부족합니다' 가 뜬다."""
        tiny = breakdown("both", dap=0.009, rest=0.002)
        actions = next_actions.build(report(map50=0.95, tide=tiny))
        self.assertNotIn("both_dominant", codes(actions))
        self.assertIn("looks_healthy", codes(actions))


class PriorityTest(unittest.TestCase):
    def test_labels_outrank_the_error_type_prescription(self) -> None:
        """라벨이 틀린 채로 해상도를 올리면 틀린 것을 더 정확히 배운다."""
        actions = next_actions.build(
            report(
                tide=breakdown("miss"),
                labels={"available": True, "model_evidence": True, "total": 137},
            )
        )
        self.assertEqual(codes(actions)[0], "labels_suspect")
        self.assertIn("miss_dominant", codes(actions))

    def test_label_candidates_without_model_evidence_do_not_trigger(self) -> None:
        actions = next_actions.build(
            report(
                tide=breakdown("miss"),
                labels={"available": True, "model_evidence": False, "total": 137},
            )
        )
        self.assertNotIn("labels_suspect", codes(actions))

    def test_a_handful_of_label_candidates_is_not_worth_a_headline(self) -> None:
        actions = next_actions.build(
            report(
                instances=1000,
                tide=breakdown("miss"),
                labels={"available": True, "model_evidence": True, "total": 8},
            )
        )
        self.assertNotIn("labels_suspect", codes(actions))

    def test_at_most_three_actions(self) -> None:
        actions = next_actions.build(
            report(
                conf=0.55,
                tide=breakdown("miss", dupe=400),
                labels={"available": True, "model_evidence": True, "total": 500},
                worst=({"name": "weak", "ap50_95": 0.1, "message": "x"},),
            )
        )
        self.assertEqual(len(actions), next_actions.ACTION_CAP)

    def test_weak_class_is_only_raised_when_nothing_dominates(self) -> None:
        even = breakdown("miss", dap=0.05, rest=0.045)
        weak = ({"name": "scratch", "ap50_95": 0.1, "message": "x"},)
        self.assertIn(
            "one_weak_class", codes(next_actions.build(report(tide=even, worst=weak)))
        )
        self.assertNotIn(
            "one_weak_class",
            codes(next_actions.build(report(tide=breakdown("miss"), worst=weak))),
        )


class DuplicateTest(unittest.TestCase):
    def test_noise_below_the_deployment_threshold_is_not_a_problem(self) -> None:
        """중복 457건 중 배포 임계값에서 보이는 게 2건이면 NMS 를 만질 이유가 없다."""
        noisy = breakdown("miss", dap=0.001, rest=0.001, dupe=457, dupe_seen=2)
        self.assertNotIn("dupe_notable", codes(next_actions.build(report(tide=noisy))))

    def test_duplicates_visible_at_the_deployment_threshold_are_reported(self) -> None:
        real = breakdown("miss", dap=0.001, rest=0.001, dupe=457, dupe_seen=120)
        action = [
            a for a in next_actions.build(report(tide=real)) if a["code"] == "dupe_notable"
        ][0]
        self.assertIn("120", action["cause"])

    def test_an_old_report_without_the_threshold_count_falls_back(self) -> None:
        old = {
            "errors": [{"kind": "dupe", "dap": 0.0, "count": 300}],
            "confusion_pairs": [],
        }
        self.assertIn("dupe_notable", codes(next_actions.build(report(tide=old))))


class ConfidenceTest(unittest.TestCase):
    def test_a_threshold_far_from_the_default_is_worth_saying(self) -> None:
        actions = next_actions.build(report(conf=0.55, tide=breakdown("miss")))
        self.assertIn("conf_far_from_default", codes(actions))

    def test_a_threshold_near_the_default_is_not(self) -> None:
        actions = next_actions.build(report(conf=0.28, tide=breakdown("miss")))
        self.assertNotIn("conf_far_from_default", codes(actions))


class RobustnessTest(unittest.TestCase):
    def test_a_report_from_before_error_analysis_still_works(self) -> None:
        """schema_version 1 리포트에는 tide 도 label_issues 도 없다."""
        actions = next_actions.build(report(tide=None, labels=None))
        self.assertTrue(actions)
        self.assertNotIn("miss_dominant", codes(actions))

    def test_a_failed_breakdown_is_treated_as_absent(self) -> None:
        actions = next_actions.build(
            report(tide={"failed": True, "message": "터졌습니다"})
        )
        self.assertTrue(actions)

    def test_nothing_notable_still_returns_one_sentence(self) -> None:
        actions = next_actions.build(report(tide=breakdown("miss", dap=0.0, rest=0.0)))
        self.assertEqual(codes(actions), ["looks_healthy"])

    def test_none_everywhere_does_not_raise(self) -> None:
        actions = next_actions.build(
            report(map50=None, precision=None, recall=None, conf=None, reliable=True)
        )
        self.assertEqual(codes(actions), ["model_not_ready"])

    def test_an_empty_dictionary_does_not_raise(self) -> None:
        self.assertEqual(codes(next_actions.build({})), ["model_not_ready"])

    def test_no_placeholder_leaks_into_any_sentence(self) -> None:
        for tide in (None, breakdown("miss"), breakdown("cls"), breakdown("bkg")):
            for action in next_actions.build(report(tide=tide)):
                for field in ("title", "cause", "fix"):
                    self.assertNotIn("{", action[field])
                    self.assertNotIn("}", action[field])

    def test_every_action_is_korean_and_complete(self) -> None:
        for action in next_actions.build(report(tide=breakdown("loc"))):
            self.assertIn(action["severity"], {"critical", "warn", "info"})
            for field in ("title", "cause", "fix"):
                self.assertTrue(action[field].strip())
                self.assertFalse(action[field].isascii())


if __name__ == "__main__":
    unittest.main()
