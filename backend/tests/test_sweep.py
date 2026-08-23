"""스윕 — 한 축을 여러 값으로 한 번에 큐에 넣기.

여기서 고정하는 계약은 하나가 핵심이다: **요청 검증이 실패하면 run 이 0개 만들어진다.**
클라이언트가 POST /api/runs 를 N번 부르는 대신 서버 엔드포인트를 둔 이유가 그것이므로,
그게 깨지면 이 기능을 둘 이유가 사라진다.

만드는 도중의 실패(디스크 참 등)까지 되돌리지는 않는다 — 그건 설계가 범위 밖으로
선언한 것이라 여기서도 시험하지 않는다.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from fastapi import HTTPException

from ._support import isolate_storage, isolate_weights


class SweepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = isolate_storage()
        isolate_weights(self, ["yolo11n.pt", "yolo11s.pt"])

        from app.api import runs as runs_api
        from app.core import db
        from app.services import run_manager

        self.api = runs_api
        self.db = db

        # schedule() 은 큐에서 꺼내 진짜 워커 subprocess 를 띄운다. 검증 테스트에서
        # 학습이 실제로 시작되면 안 되므로 잠시 막는다.
        original = run_manager.schedule
        run_manager.schedule = lambda: None
        self.addCleanup(setattr, run_manager, "schedule", original)

        db.execute(
            "INSERT INTO datasets (id,name,source,origin,root,yaml_path,classes,report,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            ("ds1", "brain-tumor", "path", "x", str(self.root), str(self.root / "d.yaml"),
             "[]", "{}", 0.0),
        )

    def payload(self, **kw):
        base = {
            "dataset_id": "ds1",
            "name": "sweep",
            "devices": [],  # CPU. 이 PC 의 GPU 유무에 테스트가 기대지 않게 한다.
            "params": {"model": "yolo11n.pt", "epochs": 3, "imgsz": 320},
            "options": {},
            "axis": "imgsz",
            "values": [320, 416, 512],
        }
        base.update(kw)
        return base

    def run_count(self) -> int:
        return len(self.db.query("SELECT id FROM runs"))

    def expect_422(self, **kw) -> str:
        """422 가 나고 run 이 하나도 안 남는지 함께 본다."""
        before = self.run_count()
        with self.assertRaises(HTTPException) as caught:
            self.api.create_sweep(self.payload(**kw))
        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(self.run_count(), before, "422 인데 run 이 만들어졌다")
        return str(caught.exception.detail)

    # ── 축 ──────────────────────────────────────────────

    def test_unknown_axis_is_rejected(self) -> None:
        self.expect_422(axis="없는항목")

    def test_options_scope_cannot_be_an_axis(self) -> None:
        """tensorboard 는 UI 옵션이라 학습 인자가 아니다."""
        self.expect_422(axis="tensorboard", values=[True, False])

    def test_bool_axis_is_rejected(self) -> None:
        self.expect_422(axis="amp", values=[True, False])

    def test_epochs_axis_is_rejected_under_a_time_budget(self) -> None:
        """예산이 에폭 수를 덮어써서(trainer.py:546) 세 실행이 전부 같아진다."""
        detail = self.expect_422(
            axis="epochs",
            values=[10, 20, 30],
            params={"model": "yolo11n.pt", "time": 0.5},
        )
        self.assertIn("에폭", detail)

    def test_epochs_axis_is_fine_without_a_budget(self) -> None:
        out = self.api.create_sweep(
            self.payload(axis="epochs", values=[3, 5], params={"model": "yolo11n.pt"})
        )
        self.assertEqual(len(out["runs"]), 2)

    # ── 값 ──────────────────────────────────────────────

    def test_value_count_is_bounded(self) -> None:
        self.expect_422(values=[])
        self.expect_422(values=[320])
        self.expect_422(values=[320, 352, 384, 416, 448, 480, 512])

    def test_values_must_be_a_list(self) -> None:
        self.expect_422(values="320,416")

    def test_duplicate_values_are_rejected(self) -> None:
        """320 과 "320" 은 검증을 지나면 같아진다. 같은 실행을 두 번 넣을 이유가 없다."""
        self.expect_422(values=[320, "320", 416])

    def test_negative_zero_is_the_same_value(self) -> None:
        """-0.0 과 0.0 은 같은 설정이다. str() 로 비교하던 동안 둘 다 큐에 들어갔다."""
        self.expect_422(axis="mosaic", values=[-0.0, 0.0])

    def test_one_bad_value_creates_nothing(self) -> None:
        """이 엔드포인트가 존재하는 이유. 32 는 imgsz 최소값 미만이다."""
        self.expect_422(values=[320, 416, 8])

    # ── 모델 축 (Codex critical 2 의 회귀) ──────────────

    def test_model_axis_resolves_every_value_before_creating(self) -> None:
        self.expect_422(
            axis="model", values=["yolo11n.pt", "없는가중치.pt"]
        )

    def test_model_axis_works_when_all_values_exist(self) -> None:
        out = self.api.create_sweep(
            self.payload(axis="model", values=["yolo11n.pt", "yolo11s.pt"])
        )
        self.assertEqual(len(out["runs"]), 2)
        self.assertEqual(
            [r["name"] for r in out["runs"]],
            ["sweep/model=yolo11n.pt", "sweep/model=yolo11s.pt"],
        )

    def test_missing_model_key_does_not_crash(self) -> None:
        """params 에 model 이 없으면 기본값이 병합돼야 한다. 없으면 KeyError -> 500."""
        out = self.api.create_sweep(self.payload(params={"epochs": 3}))
        self.assertEqual(len(out["runs"]), 3)

    # ── 정상 경로 ───────────────────────────────────────

    def test_creates_one_run_per_value(self) -> None:
        out = self.api.create_sweep(self.payload())
        runs = out["runs"]
        self.assertEqual(len(runs), 3)
        self.assertEqual(
            [r["name"] for r in runs],
            ["sweep/imgsz=320", "sweep/imgsz=416", "sweep/imgsz=512"],
        )
        self.assertEqual([r["params"]["imgsz"] for r in runs], [320, 416, 512])
        self.assertTrue(all(r["status"] == "queued" for r in runs))

    def test_only_the_axis_differs(self) -> None:
        """축 말고는 같아야 한다. 그래야 결과 차이를 그 축 탓으로 돌릴 수 있다.

        model 은 빼고 본다 — create_run 이 가중치를 run 폴더 안으로 복사하고 그 경로를
        params 에 적기 때문에(run_manager.py:105) run 마다 다를 수밖에 없다.
        스윕이 만든 차이가 아니라 원래 그런 값이다. 대신 파일 이름은 같아야 한다.
        """
        runs = self.api.create_sweep(self.payload())["runs"]
        first = runs[0]["params"]
        for other in runs[1:]:
            differing = [
                k for k in first if k != "model" and first[k] != other["params"][k]
            ]
            self.assertEqual(differing, ["imgsz"], f"축 말고도 달라졌다: {differing}")
            self.assertEqual(
                Path(str(first["model"])).name,
                Path(str(other["params"]["model"])).name,
            )

    def test_axis_label_is_returned_for_the_ui(self) -> None:
        out = self.api.create_sweep(self.payload())
        self.assertEqual(out["axis"], "imgsz")
        self.assertTrue(out["label"])


class SourceModelTest(unittest.TestCase):
    """비교 화면이 모델을 대조할 근거.

    params["model"] 은 run 폴더 안의 복사본이라 run 마다 다르다(run_manager.py:105).
    그대로 두면 어떤 두 실행을 비교해도 model 이 늘 "다른 설정" 으로 뜬다 —
    스윕에서는 "축만 달라야 한다" 는 신호가 그만큼 흐려진다.
    """

    def setUp(self) -> None:
        self.root = isolate_storage()
        self.weights = isolate_weights(self, ["yolo11n.pt", "yolo11s.pt"])

        from app.api import runs as runs_api
        from app.core import db
        from app.services import run_manager

        self.api = runs_api
        original = run_manager.schedule
        run_manager.schedule = lambda: None
        self.addCleanup(setattr, run_manager, "schedule", original)

        db.execute(
            "INSERT INTO datasets (id,name,source,origin,root,yaml_path,classes,report,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            ("ds1", "d", "path", "x", str(self.root), str(self.root / "d.yaml"),
             "[]", "{}", 0.0),
        )

    def make(self, model: str) -> dict:
        run = self.api.create_run({
            "dataset_id": "ds1", "name": "r", "devices": [],
            "params": {"model": model, "epochs": 3}, "options": {},
        })
        return self.api.get_run(run["id"])

    def test_same_weights_share_a_source_even_though_params_differ(self) -> None:
        a, b = self.make("yolo11n.pt"), self.make("yolo11n.pt")
        # params 는 반드시 다르다 - 각자 자기 run 폴더 안의 복사본을 가리킨다.
        self.assertNotEqual(a["params"]["model"], b["params"]["model"])
        # 그런데 사용자가 고른 것은 같다. 비교 화면은 이걸 봐야 한다.
        self.assertEqual(a["source_model"], b["source_model"])

    def test_different_weights_still_differ(self) -> None:
        """파일 이름으로 비교했다면 놓쳤을 구분이 살아 있는지."""
        a, b = self.make("yolo11n.pt"), self.make("yolo11s.pt")
        self.assertNotEqual(a["source_model"], b["source_model"])

    def test_same_basename_from_different_places_still_differ(self) -> None:
        """서로 다른 실행에서 이어받은 best.pt 두 개. 이름은 같고 실체는 다르다.

        이게 "model 을 파일 이름으로 비교하자" 를 기각한 이유다.
        """
        from pathlib import Path

        one, two = self.root / "a", self.root / "b"
        for d in (one, two):
            d.mkdir(parents=True, exist_ok=True)
            (d / "best.pt").write_bytes(b"")
        a, b = self.make(str(one / "best.pt")), self.make(str(two / "best.pt"))
        self.assertEqual(
            Path(str(a["source_model"])).name, Path(str(b["source_model"])).name
        )
        self.assertNotEqual(a["source_model"], b["source_model"])

    def test_missing_config_is_none_not_a_crash(self) -> None:
        """옛 run 이나 폴더가 지워진 run 도 상세를 열 수 있어야 한다."""
        from app.core import db

        db.execute(
            "INSERT INTO runs (id,name,dataset_id,status,params,options,devices,created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            ("old", "옛 run", "ds1", "completed", "{}", "{}", "[]", 0.0),
        )
        self.assertIsNone(self.api.get_run("old")["source_model"])


if __name__ == "__main__":
    unittest.main()
