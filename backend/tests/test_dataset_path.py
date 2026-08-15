"""등록된 원본 폴더가 사라졌을 때 사유가 나오는가.

경로 참조 데이터셋의 폴더를 옮기면 **사진만** 조용히 전부 깨진다 — 학습과 분석은
train.txt/val.txt 를 읽으므로 멀쩡히 돌아간다. 화면에 사유가 없으면 원인을 알 수 없다.
실제로 겪은 사고라 회귀를 막는다.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests._support import isolate_storage

from app.services import dataset_ingest  # noqa: E402


class PathStatusTest(unittest.TestCase):
    def setUp(self):
        self.root = isolate_storage()
        self.images = self.root / "src" / "images"
        self.images.mkdir(parents=True)
        self.image = self.images / "a.jpg"
        self.image.write_bytes(b"not really a jpeg")

        self.dataset_dir = dataset_ingest.DATASETS_DIR / "ds1"
        self.dataset_dir.mkdir(parents=True)
        self.dataset = {"id": "ds1", "root": str(self.root / "src")}

    def _write_list(self, *paths: Path) -> None:
        (self.dataset_dir / "train.txt").write_text(
            "\n".join(str(p) for p in paths) + "\n", encoding="utf-8"
        )

    def test_healthy_dataset_is_ok(self):
        self._write_list(self.image)
        status = dataset_ingest.path_status(self.dataset)
        self.assertTrue(status["ok"])
        self.assertEqual(status["code"], "ok")

    def test_missing_root_names_the_path(self):
        self._write_list(self.image)
        status = dataset_ingest.path_status(
            {"id": "ds1", "root": str(self.root / "gone")}
        )
        self.assertFalse(status["ok"])
        self.assertEqual(status["code"], "root_missing")
        self.assertIn("gone", status["message"])

    def test_missing_list_file(self):
        status = dataset_ingest.path_status(self.dataset)
        self.assertEqual(status["code"], "list_missing")

    def test_empty_list_file(self):
        (self.dataset_dir / "train.txt").write_text("\n", encoding="utf-8")
        self.assertEqual(
            dataset_ingest.path_status(self.dataset)["code"], "list_missing"
        )

    def test_images_missing(self):
        self._write_list(*[self.images / f"gone{k}.jpg" for k in range(30)])
        status = dataset_ingest.path_status(self.dataset)
        self.assertEqual(status["code"], "images_missing")

    def test_probe_stops_after_the_sample_limit(self):
        """전수로 보면 목록 API 가 데이터셋마다 이미지 수만큼 stat 을 부른다."""
        missing = [
            self.images / f"gone{k}.jpg"
            for k in range(dataset_ingest.PATH_PROBE_SAMPLE)
        ]
        self._write_list(*missing, self.image)
        self.assertEqual(
            dataset_ingest.path_status(self.dataset)["code"], "images_missing"
        )

    def test_finds_a_later_file_within_the_sample(self):
        self._write_list(self.images / "gone.jpg", self.image)
        self.assertTrue(dataset_ingest.path_status(self.dataset)["ok"])

    def test_list_pointing_outside_root(self):
        """폴더를 옮기며 목록만 고치고 등록 경로는 그대로 둔 상태 — 실제로 겪은 사고다."""
        moved = self.root / "moved" / "images"
        moved.mkdir(parents=True)
        outside = moved / "a.jpg"
        outside.write_bytes(b"not really a jpeg")
        self._write_list(outside)
        status = dataset_ingest.path_status(self.dataset)
        self.assertEqual(status["code"], "outside_root")
        self.assertIn(str(moved.resolve()), status["message"])

    def test_blank_root(self):
        self.assertEqual(
            dataset_ingest.path_status({"id": "ds1", "root": ""})["code"],
            "root_missing",
        )


if __name__ == "__main__":
    unittest.main()
