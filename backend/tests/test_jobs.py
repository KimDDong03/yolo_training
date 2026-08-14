"""사이드잡 인프라 검증.

여기서 틀리면 조용히 아프다. GPU 예약을 놓치면 학습과 잡이 같은 GPU 에 올라가 둘 다 OOM
나고, 경로 경계가 새면 디스크의 아무 파일이나 읽힌다. 그 두 가지에 집중한다.
"""

from __future__ import annotations

import json
import sqlite3
import time
import unittest

from ._support import force_worker_alive, isolate_storage


class JobsTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = isolate_storage()
        from app.core import config, db
        from app.services import jobs, run_manager

        self.config, self.db, self.jobs, self.run_manager = config, db, jobs, run_manager
        self.run_id = "run1"
        (config.RUNS_DIR / self.run_id).mkdir(parents=True, exist_ok=True)
        db.execute(
            "INSERT INTO runs (id,name,dataset_id,status,params,options,devices,retry_of,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (self.run_id, "t", "ds", "completed", "{}", "{}", "[0]", None, time.time()),
        )

    def insert_job(self, *, status: str = "running", pid: int | None = None,
                   devices: str = "[0]", kind: str = "export", owner: str | None = None) -> str:
        job_id = f"j{int(time.time() * 1e6) % 10**9}{status[:2]}{devices}{kind}"[:12]
        now = time.time()
        self.db.execute(
            "INSERT INTO jobs (id,kind,owner_type,owner_id,status,args,devices,pid,created_at,started_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job_id, kind, "run", owner or self.run_id, status, "{}", devices, pid, now, now),
        )
        return job_id


class GpuReservationTest(JobsTestBase):
    def test_dead_jobs_do_not_hold_a_gpu(self) -> None:
        """PID 가 남아 있어도 프로세스가 없으면 슬롯을 풀어야 한다."""
        self.insert_job(pid=None)
        self.assertEqual(self.jobs.reserved_devices(), set())

    def test_finished_jobs_do_not_hold_a_gpu(self) -> None:
        self.insert_job(status="completed", pid=None)
        self.assertEqual(self.jobs.reserved_devices(), set())

    def test_busy_devices_includes_jobs(self) -> None:
        """이걸 빠뜨리면 스케줄러가 잡이 쓰는 GPU 위에 학습을 띄운다."""
        import os

        self.insert_job(pid=os.getpid(), devices="[0]")
        force_worker_alive(self)
        self.assertEqual(self.jobs.reserved_devices(), {0})
        self.assertIn(0, self.run_manager.busy_devices())

    def test_training_and_jobs_share_one_slot(self) -> None:
        """프로세스를 분리해도 VRAM 은 분리되지 않는다."""
        self.db.execute("UPDATE runs SET status='running' WHERE id=?", (self.run_id,))
        self.assertEqual(self.run_manager.training_devices(), {0})
        self.assertIn(0, self.run_manager.busy_devices())


class DuplicateGuardTest(JobsTestBase):
    def test_database_rejects_two_live_jobs_of_a_kind(self) -> None:
        self.insert_job()
        with self.assertRaises(sqlite3.IntegrityError):
            self.insert_job()

    def test_finished_jobs_may_repeat(self) -> None:
        self.insert_job(status="completed")
        self.insert_job(status="failed")
        self.insert_job(status="running")  # 던지지 않아야 한다

    def test_duplicate_becomes_a_friendly_409_not_a_500(self) -> None:
        """IntegrityError 를 안 잡으면 친절한 안내가 500 으로 퇴화한다."""
        self.insert_job()
        force_worker_alive(self)
        with self.assertRaises(self.run_manager.RunError) as caught:
            self.run_manager.start_job(
                "export", "run", self.run_id,
                {"format": "onnx", "weights": "train/weights/best.pt"},
            )
        self.assertIn("내보내기", str(caught.exception))


class PathBoundaryTest(JobsTestBase):
    def test_owner_id_cannot_escape_the_storage_root(self) -> None:
        for bad in ("../..", "../other", "a/../../b"):
            with self.subTest(owner_id=bad):
                with self.assertRaises(self.jobs.JobError):
                    self.jobs.owner_dir("run", bad)

    def test_unknown_owner_type_is_rejected(self) -> None:
        with self.assertRaises(self.jobs.JobError):
            self.jobs.owner_dir("secrets", "x")

    def test_job_output_lives_under_its_owner(self) -> None:
        directory = self.jobs.job_dir("export", "run", self.run_id)
        # 소유자 폴더 아래에 두면 run/dataset 을 지울 때 산출물이 함께 사라진다.
        self.assertIn(self.run_id, directory.parts)
        self.assertEqual(directory.parts[-2:], ("jobs", "export"))

    def test_export_rejects_weights_outside_the_run(self) -> None:
        """소유자 폴더 밖의 파일을 내보내지 못하게 한다."""
        for bad in ("../../secret.pt", "C:/Windows/system32/x.pt", "/etc/passwd"):
            with self.subTest(weights=bad):
                with self.assertRaises(self.jobs.JobError):
                    self.jobs.spec_for("export").validate({"format": "onnx", "weights": bad})

    def test_export_rejects_unknown_formats(self) -> None:
        with self.assertRaises(self.jobs.JobError):
            self.jobs.spec_for("export").validate({"format": "coreml"})


class LiveForTest(JobsTestBase):
    def test_live_for_ignores_dead_jobs(self) -> None:
        self.insert_job(pid=None)
        self.assertEqual(self.jobs.live_for("run", self.run_id), [])

    def test_delete_guard_blocks_while_a_job_runs(self) -> None:
        self.insert_job(pid=1)
        force_worker_alive(self)
        with self.assertRaises(self.run_manager.RunError):
            with self.run_manager.exclusive_delete("run", self.run_id):
                self.fail("가드를 통과하면 안 된다")

    def test_delete_proceeds_when_nothing_is_running(self) -> None:
        entered = False
        with self.run_manager.exclusive_delete("run", self.run_id):
            entered = True
        self.assertTrue(entered)


class LegacyAdoptionTest(JobsTestBase):
    def test_live_legacy_export_is_adopted(self) -> None:
        """이관 시점에 돌고 있던 내보내기의 GPU 예약을 놓치면 학습과 둘 다 OOM 난다."""
        record = {"pid": 4242, "started_at": time.time(), "devices": [0], "format": "engine"}
        path = self.config.RUNS_DIR / self.run_id / "export.job.json"
        path.write_text(json.dumps(record), encoding="utf-8")

        force_worker_alive(self)
        self.jobs._adopt_legacy_exports()

        rows = self.db.query("SELECT * FROM jobs WHERE owner_id = ?", (self.run_id,))
        self.assertEqual(len(rows), 1)
        job = self.db.row_to_job(rows[0])
        self.assertEqual(job["kind"], "export")
        self.assertEqual(job["devices"], [0])
        self.assertFalse(path.exists(), "흡수한 뒤에는 구버전 파일을 남기지 않는다")

    def test_dead_legacy_export_is_only_cleaned_up(self) -> None:
        path = self.config.RUNS_DIR / self.run_id / "export.job.json"
        path.write_text(json.dumps({"pid": 999999, "started_at": 1.0, "devices": [0]}), encoding="utf-8")
        self.jobs._adopt_legacy_exports()
        self.assertEqual(self.db.query("SELECT * FROM jobs"), [])
        self.assertFalse(path.exists())

    def test_corrupt_legacy_file_does_not_crash_startup(self) -> None:
        path = self.config.RUNS_DIR / self.run_id / "export.job.json"
        path.write_text("{ 깨진 json", encoding="utf-8")
        self.jobs._adopt_legacy_exports()  # 던지면 실패
        self.assertFalse(path.exists())


class StatusShapeTest(JobsTestBase):
    def test_idle_when_never_run(self) -> None:
        state = self.jobs.status("export", "run", self.run_id)
        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["events"], [])
        self.assertIsNone(state["result"])

    def test_export_adapter_keeps_the_old_response_keys(self) -> None:
        """프론트(types.ts 의 ExportStatus)를 고치지 않고 갈아끼우는 것이 목표다."""
        from app.api.runs import _as_export

        shaped = _as_export(self.jobs.status("export", "run", self.run_id))
        for key in ("status", "events", "result", "format"):
            self.assertIn(key, shaped)

    def test_crashed_job_is_settled_as_failed(self) -> None:
        """end 이벤트도 못 남기고 죽은 잡은 running 으로 영원히 남으면 안 된다."""
        self.insert_job(pid=None)
        state = self.jobs.status("export", "run", self.run_id)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(
            self.db.query_one("SELECT status FROM jobs LIMIT 1")["status"], "failed"
        )


if __name__ == "__main__":
    unittest.main()
