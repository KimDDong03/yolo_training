"""실행 목록 요약(run_summary)이 무엇을 읽고 무엇을 다시 읽지 않는지 검증한다.

여기서 캐시를 시험하는 이유: 목록은 2초마다 폴링된다. 캐시가 깨지면 목록 요청 하나가
실행 수만큼 events.jsonl 을 통째로 읽는다. 그 회귀는 화면에 안 보이고 느려지기만 한다.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import _support  # noqa: F401  (sys.path 설정)

from app.services import run_summary


def write_events(run_dir: Path, lines: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "events.jsonl", "w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def epoch(n: int, total: int, m: float | None) -> dict:
    event = {"t": "epoch", "epoch": n, "total_epochs": total}
    if m is not None:
        event["summary"] = {"mAP50-95": m}
    return event


class SummarizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="yoloweb_summary_"))
        run_summary._CACHE.clear()

    def test_missing_file_is_all_none(self) -> None:
        self.assertEqual(run_summary.summarize(self.root / "없는run"), run_summary.EMPTY)

    def test_reads_last_epoch_and_best_map(self) -> None:
        run_dir = self.root / "r1"
        write_events(run_dir, [
            {"t": "start", "total_epochs": 100},
            epoch(1, 100, 0.31),
            epoch(2, 100, 0.62),
            epoch(3, 100, 0.58),  # 최고는 마지막이 아니라 최댓값이다
        ])
        self.assertEqual(
            run_summary.summarize(run_dir),
            {"epoch": 3, "total_epochs": 100, "best_map": 0.62},
        )

    def test_epoch_without_summary_still_advances_progress(self) -> None:
        """지표가 아직 안 붙은 에폭이 있어도 진행률은 나와야 한다."""
        run_dir = self.root / "r2"
        write_events(run_dir, [epoch(1, 50, 0.4), epoch(2, 50, None)])
        got = run_summary.summarize(run_dir)
        self.assertEqual((got["epoch"], got["total_epochs"]), (2, 50))
        self.assertEqual(got["best_map"], 0.4)

    def test_broken_lines_do_not_raise(self) -> None:
        """목록 응답이 깨진 이벤트 한 줄 때문에 실패하면 안 된다."""
        run_dir = self.root / "r3"
        run_dir.mkdir(parents=True)
        (run_dir / "events.jsonl").write_text(
            '{"t": "epoch", "epoch": 1, "total_epochs": 9, "summary": {"mAP50-95": 0.5}}\n'
            '{"t": "epoch", 잘린 줄\n'
            "\n",
            encoding="utf-8",
        )
        self.assertEqual(
            run_summary.summarize(run_dir),
            {"epoch": 1, "total_epochs": 9, "best_map": 0.5},
        )

    def test_returns_copy_so_caller_cannot_poison_cache(self) -> None:
        run_dir = self.root / "r4"
        write_events(run_dir, [epoch(7, 10, 0.2)])
        first = run_summary.summarize(run_dir)
        first["epoch"] = 999
        self.assertEqual(run_summary.summarize(run_dir)["epoch"], 7)


class CacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="yoloweb_summary_"))
        run_summary._CACHE.clear()

    def test_unchanged_file_is_not_read_again(self) -> None:
        run_dir = self.root / "r1"
        write_events(run_dir, [epoch(5, 10, 0.5)])
        run_summary.summarize(run_dir)

        calls = []
        original = run_summary._scan
        run_summary._scan = lambda path: (calls.append(path), original(path))[1]
        try:
            run_summary.summarize(run_dir)
            run_summary.summarize(run_dir)
        finally:
            run_summary._scan = original
        self.assertEqual(calls, [])

    def test_appended_file_is_read_again(self) -> None:
        """도는 실행은 파일이 자라므로 캐시가 맞으면 안 된다."""
        run_dir = self.root / "r2"
        write_events(run_dir, [epoch(1, 10, 0.1)])
        self.assertEqual(run_summary.summarize(run_dir)["epoch"], 1)

        with open(run_dir / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(epoch(2, 10, 0.3)) + "\n")
        got = run_summary.summarize(run_dir)
        self.assertEqual((got["epoch"], got["best_map"]), (2, 0.3))

    def test_cache_has_an_upper_bound(self) -> None:
        run_summary._CACHE.clear()
        for i in range(run_summary._CACHE_MAX + 5):
            run_dir = self.root / f"r{i}"
            write_events(run_dir, [epoch(1, 1, 0.1)])
            run_summary.summarize(run_dir)
        self.assertLessEqual(len(run_summary._CACHE), run_summary._CACHE_MAX)


if __name__ == "__main__":
    unittest.main()
