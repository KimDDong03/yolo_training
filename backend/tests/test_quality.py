"""데이터 품질 판정 — 중복 · train/val 누수 · 클래스 불균형, 그리고 경로 상태.

임계값은 실데이터 3개 전수 측정으로 정했다(.codex/phase-3.md). 여기 테스트는 그 규칙이
의도대로 도는지를 고정한다 — 특히 **오탐 0** 과 **유사도의 비전이성**을.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests._support import isolate_storage  # noqa: F401 - sys.path 를 잡는다

from app.services import quality  # noqa: E402


def gray(pattern: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """작은 패턴을 늘려 size 크기의 그레이 배열로 만든다."""
    reps = (size[0] // pattern.shape[0] + 1, size[1] // pattern.shape[1] + 1)
    return np.tile(pattern, reps)[: size[0], : size[1]]


class DHashTest(unittest.TestCase):
    def test_same_array_is_distance_zero(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, size=(8, 9)).astype(np.int16)
        self.assertEqual(quality.dhash_bits(a), quality.dhash_bits(a.copy()))

    def test_different_images_are_far(self):
        left = np.tile(np.array([[0, 255]], dtype=np.int16), (8, 5))[:, :9]
        right = np.tile(np.array([[255, 0]], dtype=np.int16), (8, 5))[:, :9]
        distance = int(np.bitwise_count(
            np.uint64(quality.dhash_bits(left)) ^ np.uint64(quality.dhash_bits(right))
        ))
        self.assertGreater(distance, 30)

    def test_candidate_pairs_respects_threshold(self):
        hashes = np.array([0b1111, 0b1110, 0b1 << 40], dtype=np.uint64)
        i, j, dist = quality.candidate_pairs(hashes, threshold=1)
        self.assertEqual(list(zip(i.tolist(), j.tolist())), [(0, 1)])
        self.assertEqual(dist.tolist(), [1])

    def test_candidate_pairs_handles_tiny_input(self):
        i, j, dist = quality.candidate_pairs(np.array([1], dtype=np.uint64))
        self.assertEqual(len(i), 0)
        self.assertEqual(len(j), 0)
        self.assertEqual(len(dist), 0)


class NccTest(unittest.TestCase):
    def test_identical_thumbs_score_one(self):
        rng = np.random.default_rng(1)
        thumb = rng.integers(0, 255, size=1024).astype(np.float32)
        norm, flat = quality.normalize_thumbs(np.stack([thumb, thumb.copy()]))
        self.assertFalse(flat.any())
        self.assertAlmostEqual(float(norm[0] @ norm[1]), 1.0, places=5)

    def test_flat_images_are_never_confirmed(self):
        """분산이 없는 이미지는 방향이 잡음이다. 0 벡터로 남겨 확정을 막는다."""
        blank = np.full(1024, 128.0, dtype=np.float32)
        almost = blank + np.linspace(0, 0.5, 1024, dtype=np.float32)
        norm, flat = quality.normalize_thumbs(np.stack([blank, almost]))
        self.assertTrue(flat.all())
        # 0 벡터끼리의 내적은 0 이라 어떤 임계도 넘지 못한다.
        self.assertLess(float(norm[0] @ norm[1]), quality.CONFIRM_NCC)

    def test_unrelated_thumbs_stay_below_threshold(self):
        rng = np.random.default_rng(2)
        mat = rng.integers(0, 255, size=(20, 1024)).astype(np.float32)
        norm, _ = quality.normalize_thumbs(mat)
        scores = norm @ norm.T
        off = scores[~np.eye(20, dtype=bool)]
        self.assertLess(float(off.max()), quality.CONFIRM_NCC)


class CenteredCosineTest(unittest.TestCase):
    def test_centering_separates_what_raw_cosine_cannot(self):
        """post-ReLU 임베딩은 전부 비음수라 원시 코사인이 1에 뭉친다. 평균을 빼야 갈린다."""
        rng = np.random.default_rng(3)
        base = np.full(64, 10.0, dtype=np.float32)
        mat = np.stack([base + rng.random(64).astype(np.float32) for _ in range(10)])
        mat = np.vstack([mat, mat[0:1]])  # 마지막은 첫 번째의 사본

        raw = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        raw_scores = raw @ raw.T
        others = raw_scores[0, 1:10]
        # 원시 코사인은 사본과 남남을 구분하지 못한다.
        self.assertGreater(float(others.min()), 0.99)

        centered = quality.center_embeddings(mat)
        scores = centered @ centered.T
        self.assertAlmostEqual(float(scores[0, 10]), 1.0, places=4)
        self.assertLess(float(scores[0, 1:10].max()), quality.CONFIRM_COSINE)


class ExactPairTest(unittest.TestCase):
    def test_groups_identical_digests(self):
        pairs = quality.exact_pairs([b"a", b"b", b"a", b"a", b"c"])
        self.assertEqual(pairs, {(0, 2), (0, 3), (2, 3)})

    def test_no_pairs_when_all_unique(self):
        self.assertEqual(quality.exact_pairs([b"a", b"b", b"c"]), set())


class GroupingTest(unittest.TestCase):
    def test_chain_becomes_one_group(self):
        groups = quality.union_find(4, [(0, 1), (1, 2)])
        self.assertEqual([sorted(m) for m in groups.values()], [[0, 1, 2]])

    def test_singletons_are_dropped(self):
        self.assertEqual(quality.union_find(3, []), {})

    def test_completeness_distinguishes_chain_from_clique(self):
        """유사도는 전이적이지 않다 — A~B, B~C 만으로 A~C 라고 말하면 안 된다."""
        chain = {(0, 1), (1, 2)}
        self.assertFalse(quality.is_complete([0, 1, 2], chain))
        clique = chain | {(0, 2)}
        self.assertTrue(quality.is_complete([0, 1, 2], clique))
        self.assertTrue(quality.is_complete([0, 1], chain))


class ImbalanceTest(unittest.TestCase):
    def test_counts_and_flags(self):
        result = quality.imbalance(
            ["cat", "dog", "bird", "fish"],
            train_counts={0: 500, 1: 12, 2: 40},
            val_counts={0: 100, 1: 3},
            train_images={0: 300, 1: 10, 2: 30},
            val_images={0: 60, 1: 3},
        )
        self.assertEqual(result["ratio"], round(500 / 12, 1))
        self.assertEqual(result["missing_in_train"], ["fish"])
        # bird 는 train 에 있는데 val 에 없다 -> 성능을 측정할 수 없다.
        self.assertEqual(result["missing_in_val"], ["bird"])
        self.assertEqual(result["rare_in_train"], ["dog"])
        self.assertEqual(result["classes"][0]["val_instances"], 100)
        self.assertEqual(result["classes"][3]["train_instances"], 0)

    def test_single_class_has_no_ratio(self):
        result = quality.imbalance(["only"], {0: 10}, {0: 2}, {0: 8}, {0: 2})
        self.assertIsNone(result["ratio"])


class MessageTest(unittest.TestCase):
    def test_clean_dataset_says_nothing_alarming(self):
        self.assertIn("없습니다", quality.duplicate_message(0, 0, 0))
        self.assertIn("없습니다", quality.leakage_message(0, 200, 0.0))

    def test_leak_above_threshold_warns_about_map(self):
        message = quality.leakage_message(8, 225, 0.0356)
        self.assertIn("mAP", message)
        self.assertIn("8", message)

    def test_leak_below_threshold_does_not_warn_about_map(self):
        message = quality.leakage_message(1, 400, 0.0025)
        self.assertNotIn("mAP", message)

    def test_chain_groups_are_not_offered_for_deletion(self):
        message = quality.duplicate_message(0, 0, 3)
        self.assertIn("눈으로 확인", message)
        self.assertNotIn("지워도", message)

    def test_notes_state_the_limits(self):
        lines = " ".join(quality.notes(100, 2, embedding_used=False))
        self.assertIn("train.txt", lines)
        self.assertIn("크롭", lines)
        self.assertIn("2장", lines)


class EndToEndRuleTest(unittest.TestCase):
    """워커가 쓰는 판정 순서를 그대로 재현한다 — 오탐 0 이 이 트랙의 수용 기준이다."""

    def _decide(self, hashes, thumbs, embeds, digests):
        H = np.asarray(hashes, dtype=np.uint64)
        T, _ = quality.normalize_thumbs(np.stack(thumbs))
        E = quality.center_embeddings(np.stack(embeds))
        i, j, _ = quality.candidate_pairs(H)
        exact = quality.exact_pairs(digests)
        is_exact = np.array([(int(a), int(b)) in exact for a, b in zip(i, j)], dtype=bool)
        ncc = np.einsum("ij,ij->i", T[i], T[j]) if len(i) else np.zeros(0)
        cos = np.einsum("ij,ij->i", E[i], E[j]) if len(i) else np.zeros(0)
        confirmed = is_exact | ((cos >= quality.CONFIRM_COSINE) & (ncc >= quality.CONFIRM_NCC))
        return {(int(a), int(b)) for a, b in zip(i[confirmed], j[confirmed])}

    def _make(self, seed, n):
        rng = np.random.default_rng(seed)
        thumbs = [rng.integers(0, 255, size=1024).astype(np.float32) for _ in range(n)]
        embeds = [10 + rng.random(64).astype(np.float32) for _ in range(n)]
        hashes = [quality.dhash_bits(t[:72].reshape(8, 9).astype(np.int16)) for t in thumbs]
        digests = [f"d{k}".encode() for k in range(n)]
        return hashes, thumbs, embeds, digests

    def test_unrelated_images_produce_nothing(self):
        hashes, thumbs, embeds, digests = self._make(11, 20)
        self.assertEqual(self._decide(hashes, thumbs, embeds, digests), set())

    def test_byte_identical_copy_is_always_found(self):
        """계층 0 은 오탐이 있을 수 없고, 다른 계층이 뭐라 하든 확정이다."""
        hashes, thumbs, embeds, digests = self._make(12, 8)
        hashes.append(hashes[3]); thumbs.append(thumbs[3].copy())
        embeds.append(embeds[3].copy()); digests.append(digests[3])
        self.assertIn((3, 8), self._decide(hashes, thumbs, embeds, digests))

    def test_content_copy_without_identical_bytes_is_found(self):
        """다시 저장한 사본 — 바이트는 다르지만 픽셀은 사실상 같다."""
        hashes, thumbs, embeds, digests = self._make(13, 8)
        hashes.append(hashes[5])
        thumbs.append(thumbs[5] + 0.4)          # 재압축 수준의 미세한 차이
        embeds.append(embeds[5] * 1.001)
        digests.append(b"different-bytes")
        self.assertIn((5, 8), self._decide(hashes, thumbs, embeds, digests))


class LeakageAccountingTest(unittest.TestCase):
    """오염된 val 장수는 쌍이 아니라 이미지로 센다 (한 장이 여러 쌍에 걸린다)."""

    def test_one_val_image_matching_two_train_images_counts_once(self):
        splits = ["train", "train", "val"]
        pairs = [(0, 2), (1, 2)]
        leaked = {b if splits[b] == "val" else a for a, b in pairs}
        self.assertEqual(len(leaked), 1)
        self.assertEqual(round(len(leaked) / splits.count("val"), 4), 1.0)

    def test_same_split_pairs_are_not_leakage(self):
        splits = ["train", "train", "val", "val"]
        pairs = [(0, 1), (2, 3)]
        cross = [(a, b) for a, b in pairs if splits[a] != splits[b]]
        self.assertEqual(cross, [])


class NextActionLeakTest(unittest.TestCase):
    """누수 처방은 임계 이상일 때만, 그리고 검사를 돌렸을 때만 뜬다."""

    REPORT = {
        "overall": {"map50": 0.95, "map50_95": 0.8, "precision": 0.9, "recall": 0.9,
                    "instances": 500},
        "conf_recommendation": {"reliable": True, "conf": 0.35},
        "tide": {"errors": []},
        "label_issues": {},
        "worst_classes": [],
    }

    def _codes(self, data_quality):
        from app.services import next_actions

        return [a["code"] for a in next_actions.build(dict(self.REPORT), data_quality)]

    def _quality(self, ratio, leaked=8, total=225):
        return {"leakage": {"ratio": ratio, "val_leaked": leaked, "val_total": total}}

    def test_fires_above_threshold(self):
        self.assertIn("val_leakage", self._codes(self._quality(0.02)))

    def test_silent_below_threshold(self):
        self.assertNotIn("val_leakage", self._codes(self._quality(0.005)))

    def test_silent_when_check_never_ran(self):
        """확인하지 않은 것을 없다고 말하지도, 있다고 말하지도 않는다."""
        self.assertNotIn("val_leakage", self._codes(None))

    def test_silent_when_section_failed(self):
        self.assertNotIn(
            "val_leakage",
            self._codes({"leakage": {"failed": True, "message": "계산하지 못했습니다"}}),
        )

    def test_threshold_comes_from_quality_module(self):
        """두 화면이 다른 선을 쓰면 서로를 부정한다."""
        self.assertIn("val_leakage", self._codes(self._quality(quality.LEAK_RATIO_ALERT)))


if __name__ == "__main__":
    unittest.main()
