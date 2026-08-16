"""하이퍼파라미터 자동 탐색(Phase 4) — 탐색 공간·리포트 조립·인자 검증.

이 잡은 수 시간 걸리므로 실제로 돌려 보며 고칠 수 없다. 그래서 **깨지면 몇 시간을 날리는
계약**을 여기서 고정한다:

1. 탐색 공간이 param_schema 밖으로 나가면, 다 돌린 뒤 결과를 폼에 넣지 못한다.
2. 강제 종료로 잘린 NDJSON 에서도 리포트가 나와야 한다.
3. 검증이 새면 422 여야 할 것이 500 이 된다.
4. devices_wanted 기본값이 1이 아니면 기존 잡들의 GPU 배정이 조용히 바뀐다.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._support import isolate_storage  # noqa: F401 - sys.path 를 잡는다

import tune_worker  # noqa: E402  (backend/ 는 _support 가 sys.path 에 넣는다)
from app.services import jobs, param_schema, tune  # noqa: E402


def record(iteration: int, fitness: float, **hyp) -> dict:
    """ultralytics Tuner 가 tune_results.ndjson 에 남기는 한 줄의 모양."""
    values = {key: bounds[0] for key, bounds in tune.SPACE.items()}
    values.update(hyp)
    return {
        "iteration": iteration,
        "fitness": fitness,
        "hyperparameters": values,
        "datasets": {"data": {"fitness": fitness, "metrics/mAP50(B)": fitness + 0.1}},
        "save_dirs": {"data": "/tmp/whatever"},
    }


def noise_of(stdev: float, baseline: float) -> dict:
    """확인 시도 결과. tune_worker.measure_noise 가 만드는 모양."""
    return {
        "seeds": list(tune.PROBE_SEEDS),
        "fitness": [baseline + stdev, baseline - stdev, baseline],
        "baseline_fitness": baseline,
        "stdev": stdev,
        "range": 2 * stdev,
    }


def crashed(iteration: int) -> dict:
    """학습이 죽은 시도. ultralytics 는 fitness 0.0 만 남기고 다음으로 넘어간다."""
    record_ = record(iteration, 0.0)
    record_["datasets"] = {"data": {"fitness": 0.0}}
    return record_


def ndjson(lines: list[str]) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="yoloweb_tune_"))
    (directory / "tune_results.ndjson").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return directory


def written(records: list[dict]) -> Path:
    return ndjson([json.dumps(r) for r in records])


class SpaceTest(unittest.TestCase):
    def test_space_stays_inside_param_schema(self):
        # 나가면 다 돌린 뒤 patch 가 422 로 죽는다. 실행 전에 여기서 잡는다.
        self.assertEqual(tune.schema_violations(), [])

    def test_every_key_is_a_training_param(self):
        index = param_schema.field_index()
        for key in tune.SPACE:
            self.assertIn(key, index, f"{key} 가 param_schema 에 없다")
            self.assertEqual(index[key]["scope"], "params")

    def test_best_hyperparameters_survive_validate(self):
        # 탐색이 경계값을 뱉어도 폼 관문을 통과해야 한다.
        edges = {key: bounds[1] for key, bounds in tune.SPACE.items()}
        self.assertEqual(set(param_schema.validate(edges, "params")), set(tune.SPACE))


class ReportTest(unittest.TestCase):
    def test_summarizes_baseline_best_and_gain(self):
        directory = written(
            [
                record(1, 0.500),
                record(2, 0.470),
                record(3, 0.560, lr0=0.004),
            ]
        )
        # 처방까지 보려면 흔들림이 재져 있어야 한다 — 못 쟀으면 제안을 접는다.
        report = tune.build_report(
            directory, {"iterations": 10}, noise=noise_of(0.005, 0.500)
        )

        self.assertEqual(report["iterations_done"], 3)
        self.assertEqual(report["iterations_target"], 10)
        self.assertEqual(report["baseline"]["fitness"], 0.500)
        self.assertEqual(report["best"]["i"], 3)
        self.assertAlmostEqual(report["gain"], 0.060, places=5)
        self.assertEqual(report["patch"]["lr0"], 0.004)
        # 시도 1 이 기준이므로 변화는 그 값에서 출발한다.
        self.assertEqual(
            report["items"][0]["changes"]["lr0"]["from"], tune.SPACE["lr0"][0]
        )

    def test_metrics_are_unwrapped_from_dataset_key(self):
        report = tune.build_report(written([record(1, 0.5)]), {"iterations": 3})
        self.assertEqual(report["trials"][0]["metrics"]["metrics/mAP50(B)"], 0.6)

    def test_partial_ndjson_still_reports(self):
        # 강제 종료로 마지막 줄이 잘린 경우. 그 한 줄 때문에 몇 시간치를 잃으면 안 된다.
        directory = ndjson(
            [
                json.dumps(record(1, 0.50)),
                json.dumps(record(2, 0.55)),
                '{"iteration": 3, "fitness": 0.6, "hyperpara',
            ]
        )
        report = tune.build_report(directory, {"iterations": 20})
        self.assertEqual(report["iterations_done"], 2)
        self.assertEqual(report["best"]["i"], 2)

    def test_no_results_is_not_an_error(self):
        report = tune.build_report(Path(tempfile.mkdtemp()), {"iterations": 5})
        self.assertEqual(report["iterations_done"], 0)
        self.assertFalse(report["available"])
        self.assertEqual(report["patch"], {})

    def test_gain_below_threshold_prescribes_nothing(self):
        gain = tune.MIN_ACTIONABLE_GAIN / 2
        report = tune.build_report(
            written([record(1, 0.500), record(2, 0.500 + gain)]),
            {"iterations": 10},
            noise=noise_of(0.0, 0.500),
        )
        self.assertEqual(report["patch"], {})
        self.assertEqual(report["items"], [])
        self.assertTrue(report["advisories"])

    def test_single_trial_has_no_comparison_yet(self):
        report = tune.build_report(written([record(1, 0.5)]), {"iterations": 10})
        self.assertEqual(report["patch"], {})
        self.assertTrue(report["advisories"])

    def test_eta_comes_from_measured_elapsed(self):
        # 시도 2개를 200초에 끝냈으면 시도당 100초. 20회 목표면 18회가 남아 1800초.
        directory = written([record(1, 0.5), record(2, 0.6)])
        report = tune.build_report(directory, {"iterations": 20}, elapsed_s=200.0)
        self.assertEqual(report["trial_time_s"], 100.0)
        self.assertEqual(report["eta_s"], 1800.0)

    def test_eta_ignores_trials_inherited_by_resume(self):
        # 이어하기로 3개를 물려받고 이번에 1개를 100초에 끝냈다 → 시도당 100초지 25초가 아니다.
        directory = written([record(i, 0.5) for i in range(1, 5)])
        report = tune.build_report(
            directory, {"iterations": 6}, elapsed_s=100.0, resumed=3
        )
        self.assertEqual(report["trial_time_s"], 100.0)
        self.assertEqual(report["eta_s"], 200.0)

    def test_eta_is_absent_without_measurement(self):
        report = tune.build_report(written([record(1, 0.5)]), {"iterations": 5})
        self.assertIsNone(report["eta_s"])

    def test_failed_baseline_prescribes_nothing(self):
        # 기준 시도가 죽으면 fitness 0.0 이 남는다. 그걸 기준으로 삼으면 아무 시도나
        # 대단해 보여 근거 없는 처방이 나간다.
        directory = written([crashed(1), record(2, 0.55)])
        report = tune.build_report(directory, {"iterations": 10})
        self.assertIsNone(report["baseline"])
        self.assertIsNone(report["gain"])
        self.assertEqual(report["patch"], {})
        self.assertFalse(report["available"])

    def test_failed_trials_are_excluded_but_still_listed(self):
        directory = written([record(1, 0.50), crashed(2), record(3, 0.60)])
        report = tune.build_report(directory, {"iterations": 10})
        self.assertEqual(len(report["trials"]), 3)
        self.assertFalse(report["trials"][1]["ok"])
        self.assertEqual(report["best"]["i"], 3)
        self.assertAlmostEqual(report["gain"], 0.10, places=5)
        self.assertTrue(any("실패" in line for line in report["advisories"]))

    def test_non_finite_fitness_is_treated_as_failed(self):
        # NaN 이 리포트에 들어가면 JSON.parse 가 죽고(Phase 0 의 그 버그), 비교도 조용히 어긋난다.
        directory = ndjson(
            [
                json.dumps(record(1, 0.5)),
                json.dumps(record(2, 0.6)).replace(
                    '"fitness": 0.6', '"fitness": NaN', 1
                ),
            ]
        )
        report = tune.build_report(directory, {"iterations": 10})
        self.assertFalse(report["trials"][1]["ok"])
        self.assertEqual(report["best"]["i"], 1)
        # 리포트 전체가 표준 JSON 으로 나가야 한다.
        json.loads(json.dumps(report, allow_nan=False))

    def test_non_finite_hyperparameter_is_treated_as_failed(self):
        # NaN 이 하나 섞이면 allow_nan=False 인 리포트 쓰기가 통째로 실패해 진행이 멈춘다.
        directory = ndjson(
            [
                json.dumps(record(1, 0.5)),
                json.dumps(record(2, 0.6)).replace('"lr0": 1e-05', '"lr0": NaN', 1),
            ]
        )
        report = tune.build_report(directory, {"iterations": 10})
        self.assertFalse(report["trials"][1]["ok"])
        json.loads(json.dumps(report, allow_nan=False))

    def test_measured_noise_raises_the_bar(self):
        # 실측에서 나온 바로 그 상황이다 — 시도 4개, 기준값의 시드 표준편차 0.01399,
        # 최고 상승폭 0.0189. 표준편차만 문턱으로 쓰면 통과하지만, "4개 중 최고" 를 감안하면
        # 문턱이 0.0233 이 되어 걸린다.
        directory = written([record(i, f) for i, f in
                             ((1, 0.13099), (2, 0.14300), (3, 0.14986), (4, 0.12780))])
        noise = noise_of(0.01399, 0.13099)
        report = tune.build_report(directory, {"iterations": 4}, noise=noise)
        self.assertAlmostEqual(report["gain"], 0.01887, places=5)
        self.assertAlmostEqual(report["threshold"], 0.02330, places=4)
        self.assertEqual(report["patch"], {})
        self.assertTrue(any("우연과 구분되지 않" in line for line in report["advisories"]))

    def test_gain_above_measured_noise_prescribes(self):
        directory = written([record(1, 0.500), record(2, 0.560, lr0=0.004)])
        report = tune.build_report(
            directory, {"iterations": 10}, noise=noise_of(0.005, 0.5)
        )
        self.assertEqual(report["patch"]["lr0"], 0.004)
        self.assertTrue(any("시드만 바꿔" in line for line in report["advisories"]))

    def test_more_trials_raise_the_bar(self):
        # 많이 뽑을수록 최고값은 노이즈만으로도 더 높이 올라간다. 문턱도 따라 올라가야 한다.
        self.assertLess(
            tune.actionable_threshold(0.01, 4), tune.actionable_threshold(0.01, 20)
        )
        self.assertAlmostEqual(tune.selection_factor(20), 2.448, places=3)

    def test_quiet_noise_cannot_lower_the_floor(self):
        # 흔들림이 0 으로 나와도 바닥선 아래로는 못 내려간다.
        directory = written([record(1, 0.500), record(2, 0.502)])
        report = tune.build_report(
            directory, {"iterations": 10}, noise=noise_of(0.0, 0.5)
        )
        self.assertEqual(report["threshold"], tune.MIN_ACTIONABLE_GAIN)
        self.assertEqual(report["patch"], {})

    def test_without_measured_noise_nothing_is_prescribed(self):
        """흔들림을 못 쟀으면 상승폭이 커도 처방하지 않는다.

        예전에는 바닥선(0.005)을 문턱으로 대신 썼다. 실측된 흔들림이 0.007~0.026 이라
        그 값은 무엇이든 통과시킨다 — 판정이 아니라 판정하는 시늉이었다.
        """
        directory = written([record(1, 0.500), record(2, 0.560, lr0=0.004)])
        report = tune.build_report(directory, {"iterations": 10})
        self.assertIsNone(report["noise"])
        self.assertAlmostEqual(report["gain"], 0.060, places=5)
        self.assertEqual(report["patch"], {})
        self.assertEqual(report["items"], [])
        self.assertTrue(any("재지 못했습니다" in line for line in report["advisories"]))
        # 재지 못했다는 사실은 감추지 않는다 — 숫자는 그대로 보여준다.
        self.assertTrue(report["available"])

    def test_a_made_up_threshold_is_never_used(self):
        """노이즈가 없을 때 actionable_threshold 를 부르지 않는다."""
        with self.assertRaises(TypeError):
            tune.actionable_threshold(None, 4)  # type: ignore[arg-type]

    def test_report_carries_no_job_status(self):
        # 잡 상태의 단일 원천은 JobStatus 다. 리포트가 복제하면 강제 종료 뒤 서로를 부정한다.
        report = tune.build_report(written([record(1, 0.5)]), {"iterations": 10})
        self.assertNotIn("status", report)


class ValidateTest(unittest.TestCase):
    def base(self, **over) -> dict:
        args = {"model": "yolo11n.yaml", "iterations": 20, "epochs": 10, "gpus": 1}
        args.update(over)
        return args

    def test_accepts_builtin_model(self):
        clean = jobs._validate_tune(self.base())
        self.assertEqual(clean["iterations"], 20)
        self.assertEqual(clean["gpus"], 1)
        self.assertFalse(clean["restart"])

    def test_unknown_model_is_job_error(self):
        with self.assertRaises(jobs.JobError):
            jobs._validate_tune(self.base(model="nope.pt"))

    def test_non_numeric_is_job_error_not_crash(self):
        # int("bad") 가 새어 나가면 422 가 아니라 500 이 된다.
        with self.assertRaises(jobs.JobError):
            jobs._validate_tune(self.base(iterations="bad"))

    def test_out_of_range_is_job_error(self):
        with self.assertRaises(jobs.JobError):
            jobs._validate_tune(self.base(iterations=1))
        with self.assertRaises(jobs.JobError):
            jobs._validate_tune(self.base(epochs=0))

    def test_restart_is_strictly_boolean(self):
        # bool("false") 는 True 다. 지우는 스위치에서 그 관용은 몇 시간치를 날린다.
        self.assertFalse(jobs._validate_tune(self.base(restart="false"))["restart"])
        self.assertTrue(jobs._validate_tune(self.base(restart=True))["restart"])
        with self.assertRaises(jobs.JobError):
            jobs._validate_tune(self.base(restart="아무거나"))
        with self.assertRaises(jobs.JobError):
            jobs._validate_tune(self.base(restart=1))

    def test_training_params_pass_the_form_gate(self):
        clean = jobs._validate_tune(self.base(imgsz=641, batch=8, fraction=0.5))
        self.assertEqual(clean["batch"], 8)
        self.assertEqual(clean["fraction"], 0.5)
        with self.assertRaises(jobs.JobError):
            jobs._validate_tune(self.base(imgsz=999999))


class SettingWarningTest(unittest.TestCase):
    """짧은 시도가 무엇을 잃는지는 시작 전에 말해야 한다. 실측 근거는 .codex/phase-4.md."""

    def test_reduced_data_is_warned(self):
        warnings = tune.setting_warnings(0.15, 10)
        self.assertEqual(len(warnings), 1)
        self.assertIn("15%", warnings[0])

    def test_short_epochs_is_warned(self):
        self.assertTrue(tune.setting_warnings(1.0, 3))

    def test_recommended_defaults_are_quiet(self):
        # 폼 기본값(데이터 100% · 에폭 10). 여기서 경고가 뜨면 기본값이 잘못된 것이다.
        self.assertEqual(tune.setting_warnings(1.0, 10), [])


class RepairTest(unittest.TestCase):
    """이어하기 전 NDJSON 수선.

    우리 read_results 가 관대한 것만으로는 부족하다 — 실제로 이어하기를 하는 ultralytics 는
    `json.loads(line)` 을 오류 처리 없이 돌린다. 잘린 꼬리를 남겨 두면 이어하기가 영원히 죽는다.
    """

    def test_truncated_tail_is_dropped(self):
        directory = ndjson(
            [
                json.dumps(record(1, 0.5)),
                json.dumps(record(2, 0.6)),
                '{"iteration": 3, "fitness": 0.7, "hyperpara',
            ]
        )
        self.assertEqual(tune.repair_results(directory), 1)
        # 수선 뒤에는 ultralytics 와 같은 방식(줄마다 json.loads)으로도 읽혀야 한다.
        text = (directory / "tune_results.ndjson").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.assertEqual([r["iteration"] for r in rows], [1, 2])

    def test_everything_after_a_broken_line_goes_too(self):
        # 중간 한 줄만 빼면 남은 iteration 번호와 ultralytics 가 세는 개수가 어긋난다.
        directory = ndjson(
            [
                json.dumps(record(1, 0.5)),
                "{깨진",
                json.dumps(record(3, 0.7)),
            ]
        )
        self.assertEqual(tune.repair_results(directory), 2)
        self.assertEqual(len(tune.read_results(directory)), 1)

    def test_clean_file_is_untouched(self):
        directory = written([record(1, 0.5), record(2, 0.6)])
        before = (directory / "tune_results.ndjson").read_bytes()
        self.assertEqual(tune.repair_results(directory), 0)
        self.assertEqual((directory / "tune_results.ndjson").read_bytes(), before)

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(tune.repair_results(Path(tempfile.mkdtemp())), 0)

    def test_repair_leaves_no_stray_temp_file(self):
        # 제자리 덮어쓰기 대신 임시 파일 + 교체다. 임시 파일이 남으면 안 된다.
        directory = ndjson([json.dumps(record(1, 0.5)), "{깨진"])
        tune.repair_results(directory)
        self.assertEqual(
            sorted(p.name for p in directory.iterdir()), ["tune_results.ndjson"]
        )


class WorkerTest(unittest.TestCase):
    """워커의 순수 부분. 탐색 자체는 몇 시간짜리라 여기서 돌리지 않는다."""

    def trials_dir(self, events: list[dict]) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="yoloweb_trials_"))
        (directory / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
        )
        return directory

    def test_current_trial_reports_epoch_within_the_running_trial(self):
        directory = self.trials_dir(
            [
                {"t": "start", "total_epochs": 5},
                {"t": "epoch", "epoch": 2},
                {"t": "batch", "i": 3, "n": 14},
            ]
        )
        current = tune_worker.current_trial(directory)
        self.assertEqual(
            current, {"epoch": 2, "total_epochs": 5, "batch": 3, "batch_total": 14}
        )

    def test_current_trial_is_none_between_trials(self):
        # 시도가 끝난 뒤 다음 시도가 시작되기 전. 마지막 에폭을 그대로 두면 준비 중인
        # 시간 내내 "에폭 5/5" 가 진행 중인 것처럼 보인다.
        directory = self.trials_dir(
            [
                {"t": "start", "total_epochs": 5},
                {"t": "epoch", "epoch": 5},
                {"t": "end", "status": "completed"},
            ]
        )
        self.assertIsNone(tune_worker.current_trial(directory))

    def test_current_trial_uses_only_the_latest_trial(self):
        # 에폭 번호는 시도마다 1부터 다시 시작한다. 앞 시도의 값이 새어 나오면 안 된다.
        directory = self.trials_dir(
            [
                {"t": "start", "total_epochs": 5},
                {"t": "epoch", "epoch": 5},
                {"t": "end", "status": "completed"},
                {"t": "start", "total_epochs": 5},
                {"t": "epoch", "epoch": 1},
            ]
        )
        current = tune_worker.current_trial(directory)
        assert current is not None
        self.assertEqual(current["epoch"], 1)

    def test_current_trial_is_none_without_events(self):
        self.assertIsNone(tune_worker.current_trial(Path(tempfile.mkdtemp())))

    def test_write_json_refuses_non_finite(self):
        # 브라우저 JSON.parse 가 죽는 것보다 여기서 드러나는 편이 낫다.
        path = Path(tempfile.mkdtemp()) / "tune.json"
        with self.assertRaises(ValueError):
            tune_worker.write_json(path, {"gain": float("nan")})

    def test_signature_round_trip(self):
        directory = Path(tempfile.mkdtemp())
        self.assertIsNone(tune_worker.signature_of(directory))
        (directory / tune_worker.SIGNATURE_NAME).write_text(
            json.dumps({"epochs": 10}), encoding="utf-8"
        )
        self.assertEqual(tune_worker.signature_of(directory), {"epochs": 10})

    def test_broken_signature_reads_as_missing(self):
        directory = Path(tempfile.mkdtemp())
        (directory / tune_worker.SIGNATURE_NAME).write_text("{깨진", encoding="utf-8")
        self.assertIsNone(tune_worker.signature_of(directory))


class SpecTest(unittest.TestCase):
    def test_tune_is_registered_on_datasets(self):
        spec = jobs.spec_for("tune")
        self.assertEqual(spec.owner_type, "dataset")
        self.assertFalse(spec.gpu_optional)

    def test_devices_wanted_follows_the_request(self):
        spec = jobs.spec_for("tune")
        self.assertEqual(spec.devices_wanted({"gpus": 4}), 4)
        self.assertFalse(spec.needs_gpu({"gpus": 0}))

    def test_existing_specs_still_want_one_device(self):
        # 기본값이 바뀌면 분석·품질·내보내기의 GPU 배정이 조용히 달라진다.
        for kind in ("analyze", "quality", "export"):
            self.assertEqual(jobs.spec_for(kind).devices_wanted({}), 1)


if __name__ == "__main__":
    unittest.main()
