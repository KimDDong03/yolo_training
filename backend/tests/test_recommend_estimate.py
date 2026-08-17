"""파라미터 추천과 시간·VRAM 예측 검증.

두 모듈 다 사용자가 그대로 믿고 '적용' 을 누르는 값을 낸다. 조용히 틀리면
학습을 몇 시간 태운 뒤에야 드러나므로, 값이 유효한 범위 안에 있는지와
조건 사이의 비율이 말이 되는지를 고정해 둔다.
"""

from __future__ import annotations

import json
import time
import unittest

from ._support import isolate_storage, isolate_weights


def dataset(
    *,
    tiny: float = 0.1,
    median: float = 0.05,
    train: int = 5000,
    total: int | None = None,
    classes: dict[str, int] | None = None,
    missing: int = 0,
    box_stats: bool = True,
) -> dict:
    report = {
        "train_count": train,
        "total_images": total if total is not None else train,
        "class_instances": classes or {"a": 100},
        "issue_counts": {"missing_label": missing},
    }
    if box_stats:
        report["box_stats"] = {
            "count": 1000,
            "tiny_ratio": tiny,
            "median_area": median,
            "area": [],
            "aspect": [],
        }
    return {"id": "ds1", "name": "합성", "report": report}


class RecommendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = isolate_storage()
        from app.services import param_schema, recommend

        self.recommend, self.param_schema = recommend, param_schema
        self.defaults = param_schema.defaults_dict("params")

    def form(self, **kw) -> dict:
        return {**self.defaults, "model": "yolo11n.pt", "imgsz": 640, "epochs": 100, **kw}

    def test_legacy_dataset_is_declined_not_rescanned(self) -> None:
        """box_stats 가 없는 구버전 데이터셋에서 재스캔하면 폼 입력 중에 수 분이 걸린다."""
        started = time.time()
        result = self.recommend.recommend(dataset(box_stats=False), self.form(), [])
        self.assertFalse(result["available"])
        self.assertEqual(result["patch"], {})
        self.assertLess(time.time() - started, 1.0, "재스캔이 일어난 것으로 보인다")

    def test_tiny_objects_raise_imgsz_toward_a_target(self) -> None:
        """목표는 데이터가 정한다. 지금 값의 배수가 아니다."""
        for tiny, expected in [(0.35, 960), (0.62, 1280)]:
            with self.subTest(tiny=tiny):
                patch = self.recommend.recommend(dataset(tiny=tiny), self.form(imgsz=640), [])["patch"]
                self.assertEqual(patch["imgsz"], expected)

    def test_recommendation_converges(self) -> None:
        """제안을 적용한 뒤 다시 물으면 또 더 큰 값을 내놓으면 안 된다.

        상대 증가(현재 x1.5)로 만들면 적용할 때마다 한 단계씩 올라가는 래칫이 된다.
        """
        data = dataset(tiny=0.62)
        form = self.form(imgsz=640)
        for _ in range(5):
            patch = self.recommend.recommend(data, form, [])["patch"]
            if "imgsz" not in patch:
                break
            form = {**form, **patch}
        else:
            self.fail("제안이 수렴하지 않는다")
        self.assertEqual(form["imgsz"], 1280)

    def test_imgsz_never_exceeds_the_ceiling(self) -> None:
        result = self.recommend.recommend(dataset(tiny=0.62), self.form(imgsz=1280), [])
        self.assertNotIn("imgsz", result["patch"])

    def test_imgsz_is_always_a_multiple_of_32(self) -> None:
        """ultralytics 는 32의 배수가 아니면 조용히 올린다. 그러면 예측이 어긋난다."""
        for current in (320, 512, 640, 700, 800):
            with self.subTest(imgsz=current):
                patch = self.recommend.recommend(
                    dataset(tiny=0.62), self.form(imgsz=current), []
                )["patch"]
                self.assertEqual(patch.get("imgsz", 320) % 32, 0)

    def test_close_mosaic_stays_below_epochs(self) -> None:
        """close_mosaic 은 '마지막 N 에폭 동안 모자이크를 끈다' 는 뜻이다.

        epochs 보다 크면 모자이크가 처음부터 끝까지 꺼져, 작은 객체에 제일 도움되는
        증강을 잃는다. 빠른 테스트(에폭 3)에서 정확히 그 상황이 된다.
        """
        for epochs in (1, 3, 10, 100):
            with self.subTest(epochs=epochs):
                patch = self.recommend.recommend(
                    dataset(tiny=0.62), self.form(epochs=epochs, close_mosaic=0), []
                )["patch"]
                self.assertLess(patch.get("close_mosaic", 0), epochs)

    def test_small_dataset_gets_more_epochs(self) -> None:
        patch = self.recommend.recommend(dataset(train=120), self.form(epochs=100), [])["patch"]
        self.assertEqual(patch["epochs"], 300)

    def test_large_dataset_gets_fewer_epochs_and_cache(self) -> None:
        patch = self.recommend.recommend(dataset(train=50000), self.form(epochs=100), [])["patch"]
        self.assertEqual(patch["epochs"], 80)
        self.assertEqual(patch["cache"], "disk")

    def test_nothing_is_proposed_when_already_correct(self) -> None:
        """지금 값과 같은 것을 제안하면 '적용' 을 눌러도 아무 일이 없다."""
        result = self.recommend.recommend(dataset(tiny=0.62), self.form(imgsz=1280), [])
        self.assertNotIn("imgsz", result["patch"])

    def test_small_dataset_turns_early_stopping_on_from_both_sides(self) -> None:
        """규칙 문구가 조기 종료를 약속하므로 patch 에 실제로 들어 있어야 한다.

        두 방향을 다 잡아야 한다 — 0 은 이 레포에서 '끄기' 고(빠른 테스트 프리셋),
        ultralytics 기본값 100 은 최고점이 나온 뒤로도 100 에폭을 더 돈다.
        한쪽만 잡으면 다른 쪽 사용자는 300 에폭을 그대로 다 돌면서 화면으로는
        '조기 종료를 함께 켜므로 다 돌지 않는다' 는 말을 듣는다.
        """
        for current in (0, 100):
            with self.subTest(patience=current):
                result = self.recommend.recommend(
                    dataset(train=100), self.form(patience=current), []
                )
                self.assertEqual(result["patch"].get("patience"), 50)

    def test_early_stopping_left_alone_when_already_tight(self) -> None:
        """이미 충분히 조여 둔 값을 굳이 50 으로 밀어 올리지 않는다."""
        result = self.recommend.recommend(dataset(train=100), self.form(patience=20), [])
        self.assertNotIn("patience", result["patch"])

    def test_imbalance_is_advice_not_a_patch(self) -> None:
        """클래스 불균형은 파라미터로 풀 수 없다. 값을 바꾸는 척하면 안 된다."""
        result = self.recommend.recommend(
            dataset(classes={"screw": 6300, "washer": 18}), self.form(), []
        )
        codes = [a["code"] for a in result["advisories"]]
        self.assertIn("class_imbalance", codes)
        self.assertNotIn("class_imbalance", result["patch"])

    def test_every_patch_passes_the_schema_allowlist(self) -> None:
        """통과 못 하는 값을 제안하면 사용자가 적용한 뒤 학습 시작이 422 로 죽는다."""
        cases = [
            dataset(tiny=0.62),
            dataset(tiny=0.62, train=100),
            dataset(train=50000),
            dataset(median=0.3, tiny=0.01),
            dataset(classes={"a": 9000, "b": 3}, missing=900, total=1000),
        ]
        for index, data in enumerate(cases):
            for imgsz in (160, 640, 1280):
                with self.subTest(case=index, imgsz=imgsz):
                    patch = self.recommend.recommend(data, self.form(imgsz=imgsz), [])["patch"]
                    self.param_schema.validate(patch, "params")  # 던지면 실패


class EstimateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = isolate_storage()
        from app.services import estimate, param_schema

        self.estimate = estimate
        self.defaults = param_schema.defaults_dict("params")

    def form(self, **kw) -> dict:
        return {**self.defaults, "model": "yolo11n.pt", "imgsz": 640, "epochs": 100, **kw}

    def test_model_scale_is_read_from_the_file_name(self) -> None:
        for ref, expected in [
            ("yolo11n.pt", "n"),
            ("yolo11x.pt", "x"),
            ("yolov8m.pt", "m"),
            ("C:/w/yolo11s.pt", "s"),
            ("custom.yaml", None),
            ("", None),
        ]:
            with self.subTest(ref=ref):
                self.assertEqual(self.estimate.model_scale(ref), expected)

    def test_unknown_scale_reports_failure_instead_of_guessing(self) -> None:
        result = self.estimate.estimate(dataset(), self.form(model="custom.yaml"), [])
        self.assertFalse(result["ok"])
        self.assertIn("reason", result)

    def test_bigger_images_cost_more_time(self) -> None:
        cheap = self.estimate.estimate(dataset(), self.form(imgsz=640, batch=16), [0])
        dear = self.estimate.estimate(dataset(), self.form(imgsz=1280, batch=16), [0])
        self.assertGreater(dear["epoch_time_s"], cheap["epoch_time_s"])

    def test_cpu_is_slower_than_gpu(self) -> None:
        """CPU 와 GPU 를 구분하지 않으면 CPU 학습 시간을 수십 배 낙관하게 된다."""
        on_gpu = self.estimate.estimate(dataset(), self.form(batch=16), [0])
        on_cpu = self.estimate.estimate(dataset(), self.form(batch=16), [])
        self.assertGreater(on_cpu["epoch_time_s"], on_gpu["epoch_time_s"] * 5)

    def test_fixed_overhead_keeps_small_datasets_sane(self) -> None:
        """이미지 수에 정비례만 시키면 16장짜리 에폭이 0.002초로 나온다.

        그 값으로 보정 배수를 구하면 수백 배가 되고, 그 배수를 다른 조건에 곱하면
        16장 학습이 몇 시간으로 예측된다.
        """
        tiny_set = self.estimate.estimate(dataset(train=16), self.form(batch=8), [0])
        self.assertGreater(tiny_set["epoch_time_s"], 0.5)

    def test_batch_is_capped_by_dataset_size(self) -> None:
        """한 배치가 데이터셋 전체보다 클 수는 없다."""
        result = self.estimate.estimate(dataset(train=16), self.form(batch=-1), [0])
        self.assertLessEqual(result["batch_effective"], 16)

    def test_auto_batch_assumption_is_stated(self) -> None:
        """batch=-1 은 실행 시점에 정해진다. 가정을 안 밝히면 빗나갔을 때 신뢰를 잃는다."""
        result = self.estimate.estimate(dataset(), self.form(batch=-1), [0])
        self.assertTrue(any("자동" in line for line in result["assumptions"]))

    def test_uncalibrated_estimates_admit_it(self) -> None:
        result = self.estimate.estimate(dataset(), self.form(batch=16), [0])
        self.assertEqual(result["source"], "analytic")
        low, high = result["range_s"]
        self.assertLess(low, result["total_time_s"])
        self.assertGreater(high, result["total_time_s"])

    def test_every_scale_has_all_three_constants(self) -> None:
        """세 표를 실측으로 갈아끼울 때 한 스케일을 빠뜨리면 KeyError 로 500 이 난다.

        analytic_epoch_seconds 와 analytic_vram_gb 는 무보호 인덱싱이라 방어가 없다.
        model_scale 이 뽑을 수 있는 문자 전체를 세 표가 다 덮어야 한다.
        """
        scales = set("nsmlx")
        for name in ("MODEL_COST", "VRAM_PER_IMAGE_GB", "VRAM_BASE_GB"):
            with self.subTest(table=name):
                self.assertEqual(set(getattr(self.estimate, name)), scales)

    def test_bigger_models_cost_more_time_and_memory(self) -> None:
        """실측값으로 갈아끼울 때 오타로 순서가 뒤집히는 것을 잡는다.

        값 자체는 실측이 바꾸므로 고정하지 않는다. 고정하는 것은 **순서**다 —
        어떤 실측이 나오든 큰 모델이 작은 모델보다 싸질 수는 없다.
        """
        order = "nsmlx"
        times = [
            self.estimate.analytic_epoch_seconds(1000, 640, s, True) for s in order
        ]
        memory = [self.estimate.analytic_vram_gb(16, 640, s, True) for s in order]
        for i in range(len(order) - 1):
            with self.subTest(pair=order[i : i + 2]):
                self.assertLess(times[i], times[i + 1])
                self.assertLess(memory[i], memory[i + 1])

    def test_absurd_calibration_ratios_are_rejected(self) -> None:
        """표본이 지금 조건과 다른 영역에 있으면 배수가 폭발한다. 그걸 쓰면 안 된다."""
        from app.core import db

        db.execute(
            "INSERT INTO datasets (id,name,source,origin,root,yaml_path,classes,report,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            ("ds1", "d", "path", "o", "r", "y", "[]",
             json.dumps({"train_count": 16}), time.time()),
        )
        params = {"model": "yolo11n.pt", "imgsz": 160, "amp": True, "fraction": 1.0}
        db.execute(
            "INSERT INTO runs (id,name,dataset_id,status,params,options,devices,retry_of,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            ("r1", "n", "ds1", "completed", json.dumps(params), "{}", "[0]", None, time.time()),
        )
        run_dir = self.root / "runs" / "r1"
        run_dir.mkdir(parents=True, exist_ok=True)
        # 에폭당 900초 — 16장짜리 학습에서 나올 수 없는 값이다.
        (run_dir / "events.jsonl").write_text(
            "".join(
                json.dumps({"t": "epoch", "epoch": i, "epoch_time_s": 900.0}) + "\n"
                for i in range(4)
            ),
            encoding="utf-8",
        )
        ratio, _, source, _same_scale = self.estimate._calibration("n", True)
        self.assertEqual(source, "analytic")
        self.assertEqual(ratio, 1.0)


class DefaultModelTest(unittest.TestCase):
    """폼에 처음 채워지는 모델은 번들에 무엇이 더 있든 가장 작은 것이어야 한다.

    추천·예측 테스트와 같은 파일에 둔 이유: 이 불변식이 깨지면 estimate 의 스케일 상수도
    recommend 의 판단도 전부 다른 모델 기준이 된다. 지키는 것은 KNOWN_MODELS 의 **선언 순서**라,
    누가 default_model 에 sorted() 를 끼워 넣으면 기본값이 yolo11m 으로 조용히 뒤집힌다 —
    그 뒤로 모든 신규 사용자의 첫 학습이 4배 느려지는데 아무도 바꾼 줄 모른다.
    """

    def setUp(self) -> None:
        isolate_storage()
        from app.services import param_schema

        self.param_schema = param_schema

    def test_smallest_model_wins_even_when_bigger_ones_exist(self) -> None:
        isolate_weights(self, ["yolo11m.pt", "yolo11n.pt", "yolo11s.pt", "yolo26n.pt"])
        self.assertTrue(self.param_schema.default_model().endswith("yolo11n.pt"))

    def test_falls_back_to_any_pt_when_none_are_known(self) -> None:
        isolate_weights(self, ["custom.pt"])
        self.assertTrue(self.param_schema.default_model().endswith("custom.pt"))

    def test_falls_back_to_a_definition_when_the_folder_is_empty(self) -> None:
        isolate_weights(self, [])
        self.assertTrue(self.param_schema.default_model().endswith(".yaml"))


if __name__ == "__main__":
    unittest.main()
