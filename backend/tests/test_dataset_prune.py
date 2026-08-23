"""원본 파일까지 지우는 경로를 고정한다.

되돌릴 수 없는 동작이라, 여기서 확인하는 것의 대부분은 "지웠는가" 가 아니라
**"지우지 않았는가"** 다. 거절해야 하는 요청 하나가 통과하면 사용자의 원본이 사라진다.
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

from tests._support import isolate_storage

from app.core import db  # noqa: E402
from app.services import dataset_prune, jobs  # noqa: E402

SPLITS = {"a": "train", "b": "train", "c": "train", "d": "val", "e": "val"}


class PruneTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = isolate_storage()
        self.base = self.tmp / "src"
        self.images = self.base / "images"
        self.labels = self.base / "labels"
        for name, split in SPLITS.items():
            image = self.images / split / f"{name}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{name}".encode())
            label = self.labels / split / f"{name}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # -------------------------------------------------------------- 만들기

    def build(self, root: Path | None = None) -> dict:
        """데이터셋 폴더·DB 행·목록 파일을 만들고 dataset dict 를 돌려준다."""
        root = root or self.base
        self.root = root
        self.dataset_dir = dataset_prune.DATASETS_DIR / "ds1"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.write_lists()

        report = {"total_images": 5, "train_count": 3, "val_count": 2}
        db.execute(
            "INSERT INTO datasets"
            " (id,name,source,origin,root,yaml_path,classes,report,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "ds1",
                "테스트",
                "path",
                str(root),
                str(root.resolve()),
                str(self.dataset_dir / "data.yaml"),
                json.dumps(["obj"]),
                json.dumps(report),
                time.time(),
            ),
        )
        return self.stored()

    def stored(self) -> dict:
        row = db.query_one("SELECT * FROM datasets WHERE id = ?", ("ds1",))
        assert row is not None
        return db.row_to_dataset(row)

    def write_lists(self) -> None:
        for split in ("train", "val"):
            listed = [
                str((self.images / split / f"{n}.jpg").resolve())
                for n, s in SPLITS.items()
                if s == split
            ]
            (self.dataset_dir / f"{split}.txt").write_text(
                "\n".join(listed) + "\n", encoding="utf-8"
            )

    def image(self, name: str) -> Path:
        return (self.images / SPLITS[name] / f"{name}.jpg").resolve()

    def label(self, name: str) -> Path:
        return (self.labels / SPLITS[name] / f"{name}.txt").resolve()

    def write_quality(self, names: list[str] | None = None) -> Path:
        """품질 리포트를 심는다. 여기 실린 경로만 삭제 후보가 된다."""
        names = list(SPLITS) if names is None else names
        directory = jobs.job_dir("quality", "dataset", "ds1")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "quality.json"
        path.write_text(
            json.dumps(
                {
                    "duplicates": {
                        "groups": [
                            {
                                "size": len(names),
                                "kind": "exact",
                                "images": [
                                    {"path": str(self.image(n)), "split": SPLITS[n]}
                                    for n in names
                                ],
                            }
                        ]
                    },
                    "leakage": {"pairs": []},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def listed(self, split: str) -> list[str]:
        return [
            line.strip()
            for line in (self.dataset_dir / f"{split}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]


class DeleteTest(PruneTestBase):
    def test_deletes_image_label_and_list_entry(self):
        dataset = self.build()
        self.write_quality()

        result = dataset_prune.delete_images(dataset, [str(self.image("b"))])

        self.assertEqual(result["deleted"], 1)
        self.assertEqual(result["failed"], [])
        self.assertFalse(self.image("b").exists())
        self.assertFalse(self.label("b").exists(), "라벨도 함께 지워야 한다")
        self.assertTrue(self.image("a").exists(), "고르지 않은 것은 그대로여야 한다")
        self.assertEqual(len(self.listed("train")), 2)
        self.assertEqual(result["train_count"], 2)
        self.assertEqual(result["val_count"], 2)

    def test_deletes_from_both_lists(self):
        """train 과 val 을 한 번에 고를 수 있다."""
        dataset = self.build()
        self.write_quality()

        dataset_prune.delete_images(
            dataset, [str(self.image("b")), str(self.image("d"))]
        )

        self.assertEqual(len(self.listed("train")), 2)
        self.assertEqual(len(self.listed("val")), 1)

    def test_ledger_records_what_was_deleted(self):
        dataset = self.build()
        self.write_quality()

        dataset_prune.delete_images(dataset, [str(self.image("b"))])

        ledger = json.loads(
            (self.dataset_dir / dataset_prune.LEDGER_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(len(ledger), 1)
        self.assertTrue(ledger[0]["ok"])
        self.assertEqual(ledger[0]["split"], "train")
        self.assertEqual(Path(ledger[0]["path"]), self.image("b"))
        self.assertEqual(Path(ledger[0]["label"]), self.label("b"))

    def test_report_is_recounted_not_just_decremented(self):
        """장수만 고치면 class_instances·issue_counts 가 조용히 낡는다."""
        dataset = self.build()
        self.write_quality()

        dataset_prune.delete_images(dataset, [str(self.image("b"))])

        stored = self.stored()["report"]
        self.assertEqual(stored["train_count"], 2)
        self.assertEqual(stored["val_count"], 2)
        self.assertEqual(stored["total_images"], 4)
        self.assertEqual(stored["class_instances"], {"obj": 4})
        self.assertIn("issue_counts", stored)

    def test_review_rebuilt_report_dropped_cache_kept(self):
        dataset = self.build()
        quality = self.write_quality()
        cache = quality.parent / "cache.npz"
        cache.write_bytes(b"cached")

        dataset_prune.delete_images(dataset, [str(self.image("b"))])

        self.assertFalse(quality.exists(), "지운 사진이 실린 리포트는 남기지 않는다")
        self.assertTrue(cache.exists(), "캐시는 남아야 재검사가 빠르다")
        review = json.loads(
            (self.dataset_dir / "review.json").read_text(encoding="utf-8")
        )
        self.assertEqual(review["total_images"], 4)


class RefusalTest(PruneTestBase):
    def test_rejects_path_not_in_report(self):
        """목록에 실려 있어도 품질 검사가 지목하지 않았으면 못 지운다."""
        dataset = self.build()
        self.write_quality(["a"])

        with self.assertRaises(dataset_prune.PruneError) as caught:
            dataset_prune.delete_images(dataset, [str(self.image("b"))])

        self.assertEqual(caught.exception.status, 422)
        self.assertTrue(self.image("b").exists())

    def test_rejects_when_report_missing(self):
        dataset = self.build()

        with self.assertRaises(dataset_prune.PruneError) as caught:
            dataset_prune.delete_images(dataset, [str(self.image("b"))])

        self.assertEqual(caught.exception.status, 409)
        self.assertTrue(self.image("b").exists())

    def test_rejects_path_outside_root(self):
        dataset = self.build()
        self.write_quality()
        outsider = self.tmp / "outside.jpg"
        outsider.write_bytes(b"outside")

        for raw in (str(outsider), "../outside.jpg"):
            with self.assertRaises(dataset_prune.PruneError) as caught:
                dataset_prune.delete_images(dataset, [raw])
            self.assertEqual(caught.exception.status, 422)

        self.assertTrue(outsider.exists())

    def test_rejects_path_not_listed(self):
        """루트 안이라는 것만으로는 부족하다."""
        dataset = self.build()
        stray = self.images / "train" / "stray.jpg"
        stray.write_bytes(b"stray")
        # 리포트에는 실어 둔다 — 걸러지는 이유가 목록 부재임을 분명히 한다.
        directory = jobs.job_dir("quality", "dataset", "ds1")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "quality.json").write_text(
            json.dumps(
                {
                    "duplicates": {
                        "groups": [
                            {
                                "size": 1,
                                "kind": "exact",
                                "images": [{"path": str(stray), "split": "train"}],
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(dataset_prune.PruneError) as caught:
            dataset_prune.delete_images(dataset, [str(stray)])

        self.assertEqual(caught.exception.status, 422)
        self.assertTrue(stray.exists())

    def test_rejects_emptying_a_split(self):
        dataset = self.build()
        self.write_quality()

        with self.assertRaises(dataset_prune.PruneError) as caught:
            dataset_prune.delete_images(
                dataset, [str(self.image(n)) for n in ("a", "b", "c")]
            )

        self.assertEqual(caught.exception.status, 409)
        for name in ("a", "b", "c"):
            self.assertTrue(self.image(name).exists(), "한 장도 지우면 안 된다")
        self.assertEqual(len(self.listed("train")), 3)

    def test_rejects_while_training(self):
        dataset = self.build()
        self.write_quality()
        db.execute(
            "INSERT INTO runs (id,name,dataset_id,status,params,devices,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            ("r1", "학습", "ds1", "running", "{}", "[]", time.time()),
        )

        with self.assertRaises(dataset_prune.PruneError) as caught:
            dataset_prune.delete_images(dataset, [str(self.image("b"))])

        self.assertEqual(caught.exception.status, 409)
        self.assertTrue(self.image("b").exists())

    def test_rejects_while_analyze_job_alive(self):
        """분석 잡은 run 소유라 exclusive_delete 그물에 안 걸리는데 목록을 읽는다."""
        dataset = self.build()
        self.write_quality()
        now = time.time()
        db.execute(
            "INSERT INTO runs (id,name,dataset_id,status,params,devices,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            ("r1", "학습", "ds1", "completed", "{}", "[]", now),
        )
        db.execute(
            "INSERT INTO jobs"
            " (id,kind,owner_type,owner_id,status,args,devices,pid,created_at,started_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("j1", "analyze", "run", "r1", "running", "{}", "[]", 1234, now, now),
        )
        original = jobs.alive
        jobs.alive = lambda job: True
        try:
            with self.assertRaises(dataset_prune.PruneError) as caught:
                dataset_prune.delete_images(dataset, [str(self.image("b"))])
        finally:
            jobs.alive = original

        self.assertEqual(caught.exception.status, 409)
        self.assertTrue(self.image("b").exists())


class PartialFailureTest(PruneTestBase):
    def test_unremovable_file_stays_in_the_list(self):
        """못 지운 파일을 목록에서 빼면 다음 검사에 안 떠서 재시도할 길이 사라진다."""
        dataset = self.build()
        self.write_quality()
        locked = self.image("b")
        original = dataset_prune.fsops.remove_file

        def fake(path: Path) -> None:
            if path == locked:
                raise OSError("다른 프로그램이 사용 중입니다")
            original(path)

        dataset_prune.fsops.remove_file = fake
        try:
            result = dataset_prune.delete_images(
                dataset, [str(self.image("a")), str(locked)]
            )
        finally:
            dataset_prune.fsops.remove_file = original

        self.assertEqual(result["deleted"], 1)
        self.assertEqual(len(result["failed"]), 1)
        self.assertTrue(locked.exists())
        self.assertIn(str(locked), self.listed("train"))
        self.assertNotIn(str(self.image("a")), self.listed("train"))

    def test_label_outside_root_is_left_alone(self):
        """root 가 .../images 면 _label_for 가 root 밖(.../labels)을 가리킨다."""
        dataset = self.build(root=self.images)
        self.write_quality()

        result = dataset_prune.delete_images(dataset, [str(self.image("b"))])

        self.assertEqual(result["deleted"], 1)
        self.assertFalse(self.image("b").exists())
        self.assertTrue(self.label("b").exists(), "폴더 밖 라벨은 지우지 않는다")
        self.assertEqual(len(result["failed"]), 1)
        self.assertIn("폴더 밖", result["failed"][0]["error"])


class LedgerTest(PruneTestBase):
    def test_write_failure_is_reported_not_raised(self):
        """파일은 이미 사라졌다. 여기서 예외를 내면 지웠는데 '실패' 라고 답하게 된다."""
        dataset = self.build()
        self.write_quality()
        original = dataset_prune._write_json

        def fake(path: Path, payload) -> None:
            if path.name == dataset_prune.LEDGER_NAME:
                raise OSError("디스크가 가득 찼습니다")
            original(path, payload)

        dataset_prune._write_json = fake
        try:
            result = dataset_prune.delete_images(dataset, [str(self.image("b"))])
        finally:
            dataset_prune._write_json = original

        self.assertEqual(result["deleted"], 1)
        self.assertFalse(self.image("b").exists())
        self.assertTrue(
            any(dataset_prune.LEDGER_NAME in f["path"] for f in result["failed"]),
            f"원장 실패가 응답에 없다: {result['failed']}",
        )

    def test_corrupt_ledger_is_kept_aside_not_overwritten(self):
        """덮어쓰면 지난 삭제 기록까지 함께 사라진다."""
        dataset = self.build()
        self.write_quality()
        ledger = self.dataset_dir / dataset_prune.LEDGER_NAME
        ledger.write_text("{이건 JSON 이 아니다", encoding="utf-8")

        dataset_prune.delete_images(dataset, [str(self.image("b"))])

        kept = list(self.dataset_dir.glob(dataset_prune.LEDGER_NAME + ".corrupt.*"))
        self.assertEqual(len(kept), 1, "깨진 원장을 남겨야 한다")
        self.assertIn("이건 JSON 이 아니다", kept[0].read_text(encoding="utf-8"))
        fresh = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(len(fresh), 1)


class RecoverTest(PruneTestBase):
    def test_drops_paths_whose_files_are_already_gone(self):
        """파일은 없는데 목록에 남은 상태 — 학습이 이걸로 죽는다."""
        self.build()
        gone = self.image("b")
        gone.unlink()
        (self.dataset_dir / dataset_prune.PENDING_NAME).write_text(
            json.dumps({"paths": [str(gone)], "started_at": time.time()}),
            encoding="utf-8",
        )

        dataset_prune.recover()

        self.assertNotIn(str(gone), self.listed("train"))
        self.assertEqual(len(self.listed("train")), 2)
        self.assertFalse((self.dataset_dir / dataset_prune.PENDING_NAME).exists())

    def test_does_not_record_the_same_deletion_twice(self):
        """원장을 쓴 직후에 죽으면 다음 기동의 복구가 같은 삭제를 또 적는다."""
        dataset = self.build()
        self.write_quality()
        dataset_prune.delete_images(dataset, [str(self.image("b"))])

        # 원장은 남았는데 저널을 못 지운 상태를 만든다.
        (self.dataset_dir / dataset_prune.PENDING_NAME).write_text(
            json.dumps({"paths": [str(self.image("b"))], "started_at": time.time()}),
            encoding="utf-8",
        )
        dataset_prune.recover()

        ledger = json.loads(
            (self.dataset_dir / dataset_prune.LEDGER_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(len(ledger), 1, f"원장이 중복됐다: {ledger}")

    def test_keeps_paths_whose_files_survived(self):
        """아직 있는 파일은 건드리지 않는다 — 사용자가 취소한 삭제를 되살리면 안 된다."""
        self.build()
        alive = self.image("b")
        (self.dataset_dir / dataset_prune.PENDING_NAME).write_text(
            json.dumps({"paths": [str(alive)], "started_at": time.time()}),
            encoding="utf-8",
        )

        dataset_prune.recover()

        self.assertIn(str(alive), self.listed("train"))
        self.assertTrue(alive.exists())
        self.assertFalse((self.dataset_dir / dataset_prune.PENDING_NAME).exists())


if __name__ == "__main__":
    unittest.main()
