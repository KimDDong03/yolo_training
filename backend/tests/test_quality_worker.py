"""quality_worker.py 를 실제로 실행해 quality.json 이 나오는지 본다.

test_quality.py 는 순수 판정만 본다. 워커에는 그것으로 증명되지 않는 것이 있다 —
목록 파일 읽기, 실제 라벨 파일에서 split 별 클래스 세기, 캐시 무효화, 섹션별 실패 격리,
그리고 리포트 스키마 자체. 실데이터 검증은 사람이 한 번 하고 끝나므로 여기서 고정한다.

모델(임베딩)은 쓰지 않는다 — 가중치를 못 찾으면 워커가 NCC 한 팔로 계속 돌아야 하고,
그 경로가 여기서 함께 검증된다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from tests._support import isolate_storage

from app.core.config import BACKEND_DIR  # noqa: E402

WORKER = BACKEND_DIR / "quality_worker.py"


def write_image(path: Path, pattern: list[list[int]], size: int = 64) -> None:
    """작은 패턴을 늘려 PNG 로 쓴다. 같은 패턴 = 같은 사진."""
    from PIL import Image

    image = Image.new("RGB", (len(pattern[0]), len(pattern)))
    image.putdata([(v, v, v) for row in pattern for v in row])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((size, size), Image.Resampling.NEAREST).save(path)


def checker(seed: int, w: int = 8, h: int = 8) -> list[list[int]]:
    """이미지마다 확실히 다른 무늬. 해시가 서로 멀어야 오탐 검사가 의미를 갖는다."""
    return [[(seed * 37 + x * 53 + y * 97) % 256 for x in range(w)] for y in range(h)]


class QualityWorkerTest(unittest.TestCase):
    def setUp(self):
        self.root = isolate_storage()
        self.data = self.root / "src"
        self.dataset_dir = self.root / "datasets" / "ds1"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.out = self.dataset_dir / "jobs" / "quality"

    def _label(self, image: Path, rows: list[str]) -> None:
        parts = list(image.parts)
        parts[parts.index("images")] = "labels"
        label = Path(*parts).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _build(self, *, leak: bool = True, duplicate: bool = True) -> None:
        train, val = [], []
        for k in range(6):
            p = self.data / "images" / "train" / f"t{k}.png"
            write_image(p, checker(k))
            # 클래스 0 은 train 에 많고, 클래스 1 은 두 장에만 있다.
            self._label(
                p, ["0 0.5 0.5 0.2 0.2"] + (["1 0.2 0.2 0.1 0.1"] if k < 2 else [])
            )
            train.append(p)
        for k in range(3):
            p = self.data / "images" / "val" / f"v{k}.png"
            write_image(p, checker(100 + k))
            self._label(p, ["0 0.5 0.5 0.2 0.2"])
            val.append(p)

        if duplicate:
            # train 안에서의 정확 복사 — 지워도 되는 묶음이 되어야 한다.
            dup = self.data / "images" / "train" / "t0_copy.png"
            dup.write_bytes(train[0].read_bytes())
            self._label(dup, ["0 0.5 0.5 0.2 0.2", "1 0.2 0.2 0.1 0.1"])
            train.append(dup)

        if leak:
            # val 사진 하나를 train 에도 넣는다 — 누수 1건.
            leaked = self.data / "images" / "train" / "leaked.png"
            leaked.write_bytes(val[0].read_bytes())
            self._label(leaked, ["0 0.5 0.5 0.2 0.2"])
            train.append(leaked)

        (self.dataset_dir / "train.txt").write_text(
            "\n".join(str(p) for p in train) + "\n", encoding="utf-8"
        )
        (self.dataset_dir / "val.txt").write_text(
            "\n".join(str(p) for p in val) + "\n", encoding="utf-8"
        )
        (self.dataset_dir / "data.yaml").write_text(
            "names:\n  0: box\n  1: dot\n", encoding="utf-8"
        )

    def _run(self, imgsz: int = 224) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable,
                str(WORKER),
                "--dataset-dir",
                str(self.dataset_dir),
                "--out-dir",
                str(self.out),
                "--events",
                str(self.out / "events.jsonl"),
                "--imgsz",
                str(imgsz),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        report_path = self.out / "quality.json"
        self.assertTrue(
            report_path.is_file(),
            f"quality.json 이 없다. rc={result.returncode}\n{result.stdout}\n{result.stderr}",
        )
        return json.loads(report_path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- 검출

    def test_finds_planted_duplicate_and_leak(self):
        self._build()
        report = self._run()

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["counts"]["train"], 8)
        self.assertEqual(report["counts"]["val"], 3)

        # 심은 누수 1건. 오염된 val 은 쌍이 아니라 이미지로 센다.
        leak = report["leakage"]
        self.assertEqual(leak["val_leaked"], 1)
        self.assertEqual(leak["val_total"], 3)
        self.assertEqual(leak["ratio"], round(1 / 3, 4))
        self.assertEqual(leak["exact_pairs"], 1)
        self.assertTrue(leak["pairs"][0]["exact"])
        self.assertTrue(leak["pairs"][0]["val"].endswith("v0.png"))

        # 심은 중복 1쌍 + 누수쌍도 같은 사진이라 묶음이 된다.
        dup = report["duplicates"]
        self.assertGreaterEqual(dup["wasted"], 1)
        self.assertTrue(all(g["kind"] in ("exact", "near") for g in dup["groups"]))

    def test_clean_dataset_reports_nothing(self):
        """오탐 0 이 이 트랙의 수용 기준이다."""
        self._build(leak=False, duplicate=False)
        report = self._run()
        self.assertEqual(report["leakage"]["val_leaked"], 0)
        self.assertEqual(report["leakage"]["pair_total"], 0)
        self.assertEqual(report["duplicates"]["wasted"], 0)
        self.assertEqual(report["duplicates"]["group_total"], 0)
        self.assertIn("없습니다", report["duplicates"]["message"])

    def test_counts_classes_per_split_from_real_label_files(self):
        """scan() 은 전역만 센다. split 별 분해는 워커가 라벨 파일을 직접 읽어 만든다."""
        self._build(leak=False, duplicate=False)
        report = self._run()
        rows = {c["name"]: c for c in report["imbalance"]["classes"]}
        self.assertEqual(rows["box"]["train_instances"], 6)
        self.assertEqual(rows["box"]["val_instances"], 3)
        self.assertEqual(rows["dot"]["train_instances"], 2)
        # dot 은 val 에 하나도 없다 -> 성능을 측정할 수 없다고 알려야 한다.
        self.assertEqual(rows["dot"]["val_instances"], 0)
        self.assertEqual(report["imbalance"]["missing_in_val"], ["dot"])
        # 픽스처가 작아 두 클래스 모두 RARE_INSTANCES 아래다. 규칙이 도는지만 본다.
        self.assertIn("dot", report["imbalance"]["rare_in_train"])
        self.assertEqual(report["imbalance"]["ratio"], round(6 / 2, 1))

    # ---------------------------------------------------------------- 계약

    def test_sections_fail_independently(self):
        """클래스 이름을 못 읽어도 중복·누수 결과는 살아야 한다."""
        self._build()
        (self.dataset_dir / "data.yaml").write_text("names: []\n", encoding="utf-8")
        report = self._run()
        self.assertTrue(report["imbalance"].get("failed"))
        self.assertIn("message", report["imbalance"])
        # 나머지 두 섹션은 멀쩡하다.
        self.assertNotIn("failed", report["duplicates"])
        self.assertEqual(report["leakage"]["val_leaked"], 1)

    def test_section_helper_catches_any_exception(self):
        """섹션 격리는 "실패를 미리 알고 만든 값" 이 아니라 **예외** 를 잡아야 한다.

        위 테스트는 build_imbalance 가 스스로 만든 failed 만 통과시켜도 초록이 된다.
        중복·누수 섹션이 터졌을 때 나머지가 사는지는 이쪽이 증명한다.
        """
        import quality_worker

        # 터지는 빌더를 일부러 넘긴다. 반환형이 dict 가 아닌 것도 의도다.
        result = quality_worker.section("중복 검사", lambda: 1 / 0)  # type: ignore[arg-type]
        self.assertTrue(result["failed"])
        self.assertIn("중복 검사", result["message"])

        # 인자가 어긋나도(리팩터 중 흔한 실수) 잡 전체가 죽지는 않는다.
        # 일부러 잘못 부르는 것이 이 테스트의 전부다 — 타입 검사를 맞추려고 인자를
        # 채우면 검증하려던 것이 사라진다.
        broken = quality_worker.section(
            "누수 검사", lambda: quality_worker.build_leakage()  # type: ignore[call-arg]
        )
        self.assertTrue(broken["failed"])

    def test_single_image_fails_the_whole_job(self):
        """비교할 상대가 없으면 중복도 누수도 정의되지 않는다 — 잡 실패가 맞다."""
        image = self.data / "images" / "train" / "only.png"
        write_image(image, checker(1))
        self._label(image, ["0 0.5 0.5 0.2 0.2"])
        (self.dataset_dir / "train.txt").write_text(str(image) + "\n", encoding="utf-8")
        (self.dataset_dir / "val.txt").write_text("", encoding="utf-8")
        (self.dataset_dir / "data.yaml").write_text("names:\n  0: box\n", encoding="utf-8")
        self.out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(WORKER), "--dataset-dir", str(self.dataset_dir),
             "--out-dir", str(self.out), "--events", str(self.out / "events.jsonl"),
             "--device", "cpu"],
            capture_output=True, text=True, timeout=300,
        )
        end = [json.loads(x) for x in
               (self.out / "events.jsonl").read_text(encoding="utf-8").splitlines() if x][-1]
        self.assertEqual(end["status"], "failed")
        self.assertIn("부족", end["error"])

    def test_notes_state_the_scope(self):
        self._build(leak=False, duplicate=False)
        joined = " ".join(self._run()["notes"])
        self.assertIn("train.txt", joined)
        self.assertIn("크롭", joined)

    def test_empty_lists_fail_the_whole_job_with_a_reason(self):
        """목록이 비면 검사할 것이 없다 — 섹션 격리 대상이 아니라 잡 실패다."""
        (self.dataset_dir / "train.txt").write_text("", encoding="utf-8")
        (self.dataset_dir / "val.txt").write_text("", encoding="utf-8")
        self.out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(WORKER),
                "--dataset-dir",
                str(self.dataset_dir),
                "--out-dir",
                str(self.out),
                "--events",
                str(self.out / "events.jsonl"),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        events = [
            json.loads(x)
            for x in (self.out / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if x
        ]
        end = events[-1]
        self.assertEqual(end["status"], "failed")
        self.assertIn("목록", end["error"])

    # ---------------------------------------------------------------- 캐시

    def _poison_cache(self) -> None:
        """캐시의 해시를 전부 같은 값으로 바꾼다.

        "결과가 같더라" 로는 캐시를 탔는지 알 수 없다 — 안 타도 결과는 같기 때문이다.
        캐시를 타면 반드시 달라지는 흔적을 심어 두고 그 흔적으로 판별한다.
        """
        import numpy as np

        blob = dict(np.load(self.out / "cache.npz"))
        blob["hashes"] = np.zeros_like(blob["hashes"])
        np.savez_compressed(self.out / "cache.npz", **blob)

    def test_cache_is_actually_reused_on_rerun(self):
        self._build(leak=False, duplicate=False)
        first = self._run()
        baseline = first["counts"]["candidate_pairs"]
        self.assertTrue((self.out / "cache.npz").is_file())
        self.assertEqual(
            json.loads((self.out / "cache.json").read_text(encoding="utf-8"))["imgsz"], 224
        )

        # 해시를 전부 0 으로 만들었으니 캐시를 탔다면 모든 쌍이 후보가 된다.
        self._poison_cache()
        second = self._run()
        n = second["counts"]["scanned"]
        self.assertEqual(second["counts"]["candidate_pairs"], n * (n - 1) // 2)
        self.assertGreater(second["counts"]["candidate_pairs"], baseline)

    def test_cache_is_dropped_when_imgsz_changes(self):
        """224 로 만든 임베딩을 640 실행에 재사용하면 조용히 틀린 판정이 나온다."""
        self._build(leak=False, duplicate=False)
        baseline = self._run(imgsz=224)["counts"]["candidate_pairs"]
        self._poison_cache()

        # imgsz 가 다르면 캐시를 버려야 하므로, 심은 흔적이 결과에 나타나면 안 된다.
        report = self._run(imgsz=640)
        self.assertEqual(report["params"]["imgsz"], 640)
        self.assertEqual(report["counts"]["candidate_pairs"], baseline)
        self.assertEqual(
            json.loads((self.out / "cache.json").read_text(encoding="utf-8"))["imgsz"], 640
        )


if __name__ == "__main__":
    unittest.main()
