"""실패 진단 규칙표 검증.

규칙은 조용히 낡는다 — 로그 문구가 바뀌거나 정규식을 고치다 다른 규칙을 가려도
아무도 모른다. 각 규칙이 자기 신호에만 걸리는지, 그리고 처방이 실제로 유효한 값을
만드는지를 고정해 둔다.
"""

from __future__ import annotations

import json
import time
import unittest

from ._support import isolate_storage


class DiagnoseTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = isolate_storage()
        from app.core import config, db
        from app.services import diagnose_fail

        self.config, self.db, self.diagnose_fail = config, db, diagnose_fail
        self.counter = 0

    def make_run(
        self,
        log: str,
        *,
        error: str | None = None,
        params: dict | None = None,
        devices: list[int] | None = None,
        status: str = "failed",
    ) -> str:
        self.counter += 1
        run_id = f"run{self.counter}"
        run_dir = self.config.RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        params = params or {
            "model": str(run_dir / "inputs" / "yolo11n.pt"),
            "batch": 16,
            "amp": False,
            "lr0": 0.01,
            "workers": 8,
        }
        devices = [0] if devices is None else devices
        (run_dir / "train.log").write_text(log, encoding="utf-8")
        (run_dir / "config.json").write_text(
            json.dumps(
                {"source_model": "C:/bundle/weights/yolo11n.pt", "params": params}
            ),
            encoding="utf-8",
        )
        (run_dir / "events.jsonl").write_text(
            json.dumps({"t": "end", "status": status, "error": error}) + "\n",
            encoding="utf-8",
        )
        self.db.execute(
            "INSERT INTO runs"
            " (id,name,dataset_id,status,params,options,devices,retry_of,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                "t",
                "ds",
                status,
                json.dumps(params),
                "{}",
                json.dumps(devices),
                None,
                time.time(),
            ),
        )
        return run_id


class RuleMatchingTest(DiagnoseTestBase):
    CASES = [
        (
            "cuda_oom",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        ),
        ("nan_loss", "WARNING NaN or Inf found in input tensor."),
        ("no_labels", "train: WARNING No labels found in D:/data/labels"),
        ("data_missing", "데이터셋 정의를 찾을 수 없습니다: D:/data/data.yaml"),
        (
            "download_blocked",
            "Downloading https://github.com/ultralytics/assets/yolo11n.pt",
        ),
        ("ddp_failed", "torch.distributed.DistBackendError: NCCL error"),
        (
            "worker_crash",
            "RuntimeError: DataLoader worker (pid 123) is killed by signal",
        ),
        ("disk_full", "OSError: [Errno 28] No space left on device"),
    ]

    def test_each_signal_hits_its_own_rule(self) -> None:
        for code, log in self.CASES:
            with self.subTest(code=code):
                result = self.diagnose_fail.diagnose(self.make_run(log))
                self.assertTrue(result["matched"])
                self.assertEqual(result["code"], code)
                self.assertTrue(result["title"] and result["fix"])

    def test_unknown_failure_still_reports_something(self) -> None:
        result = self.diagnose_fail.diagnose(
            self.make_run("그냥 알 수 없는 로그", error="무언가 이상한 오류")
        )
        self.assertFalse(result["matched"])
        self.assertIsNone(result["retry"])
        self.assertIn("무언가 이상한 오류", result["cause"])

    def test_healthy_run_is_not_diagnosed(self) -> None:
        run_id = self.make_run("CUDA out of memory", status="completed")
        result = self.diagnose_fail.diagnose(run_id)
        self.assertFalse(result["matched"])
        self.assertIsNone(result["retry"])

    def test_error_only_in_train_log_is_found(self) -> None:
        """워커가 end 이벤트를 남기지 못하고 죽으면 진짜 메시지가 train.log 에만 있다."""
        run_id = self.make_run("CUDA out of memory", error=None)
        self.assertEqual(self.diagnose_fail.diagnose(run_id)["code"], "cuda_oom")


class RetryPatchTest(DiagnoseTestBase):
    def test_oom_halves_batch_and_enables_amp(self) -> None:
        result = self.diagnose_fail.diagnose(self.make_run("CUDA out of memory"))
        retry = result["retry"]
        self.assertEqual(retry["params"]["batch"], 8)
        self.assertTrue(retry["params"]["amp"])
        self.assertEqual(retry["changed"]["batch"], {"from": 16, "to": 8})

    def test_oom_with_auto_batch_picks_a_concrete_value(self) -> None:
        """batch=-1 로 이미 터졌다면 자동 산정이 과했다는 뜻이다. -1 을 반으로 나눌 수는 없다."""
        params = {"model": "m.pt", "batch": -1, "amp": True}
        result = self.diagnose_fail.diagnose(
            self.make_run("CUDA out of memory", params=params)
        )
        self.assertEqual(result["retry"]["params"]["batch"], 8)

    def test_oom_never_produces_batch_below_one(self) -> None:
        params = {"model": "m.pt", "batch": 1, "amp": True}
        result = self.diagnose_fail.diagnose(
            self.make_run("CUDA out of memory", params=params)
        )
        self.assertGreaterEqual(result["retry"]["params"]["batch"], 1)

    def test_nan_lowers_lr_and_disables_amp(self) -> None:
        params = {"model": "m.pt", "amp": True, "lr0": 0.01}
        result = self.diagnose_fail.diagnose(
            self.make_run("NaN or Inf found in input tensor.", params=params)
        )
        retry = result["retry"]
        self.assertFalse(retry["params"]["amp"])
        self.assertEqual(retry["params"]["lr0"], 0.005)

    def test_ddp_failure_shows_the_device_change(self) -> None:
        """devices 는 params 밖에 있다. 빠뜨리면 '한 장으로 재시작' 이 변경 없음으로 보인다."""
        result = self.diagnose_fail.diagnose(
            self.make_run("NCCL error", devices=[0, 1])
        )
        retry = result["retry"]
        self.assertEqual(retry["devices"], [0])
        self.assertEqual(retry["changed"]["devices"], {"from": [0, 1], "to": [0]})

    def test_worker_crash_reduces_workers(self) -> None:
        result = self.diagnose_fail.diagnose(self.make_run("DataLoader worker crashed"))
        self.assertEqual(result["retry"]["params"]["workers"], 2)

    def test_retry_restores_the_original_weights_path(self) -> None:
        """params.model 은 run 폴더 안의 사본을 가리킨다.

        그대로 재시도하면 이 실패 run 을 지우는 순간 재시도로 만든 run 도 깨진다.
        """
        result = self.diagnose_fail.diagnose(self.make_run("CUDA out of memory"))
        model = result["retry"]["params"]["model"]
        self.assertEqual(model, "C:/bundle/weights/yolo11n.pt")
        self.assertNotIn("inputs", model)
        # 내부 경로 정리는 사용자가 바꾼 설정이 아니므로 변경 목록에 뜨면 안 된다.
        self.assertNotIn("model", result["retry"]["changed"])

    def test_unfixable_failures_offer_no_retry_button(self) -> None:
        """데이터를 고치기 전에는 똑같이 실패한다. 누르면 또 실패하는 버튼은 주지 않는다."""
        for log in (
            "train: WARNING No labels found",
            "데이터셋 정의를 찾을 수 없습니다",
        ):
            with self.subTest(log=log):
                self.assertIsNone(
                    self.diagnose_fail.diagnose(self.make_run(log))["retry"]
                )


class PatchValidityTest(DiagnoseTestBase):
    def test_patched_params_pass_the_schema_allowlist(self) -> None:
        """처방이 만든 값이 param_schema 를 통과하지 못하면 재시도가 422 로 죽는다."""
        from app.services import param_schema

        for log in ("CUDA out of memory", "NaN or Inf found", "DataLoader worker"):
            with self.subTest(log=log):
                retry = self.diagnose_fail.diagnose(self.make_run(log))["retry"]
                param_schema.validate(retry["params"], "params")  # 던지면 실패


if __name__ == "__main__":
    unittest.main()
