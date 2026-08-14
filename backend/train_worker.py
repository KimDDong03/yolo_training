"""학습 워커 — 웹서버와 완전히 분리된 독립 프로세스.

진행 상황은 이 스크립트가 아니라 hooks/yoloweb_events.py 의 전역 콜백이 기록한다.
(그래야 DDP 자식 프로세스에서도 같은 코드가 돈다. hooks/sitecustomize.py 주석 참고)

사용법: python train_worker.py --run-dir <storage/runs/xxxx>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import apply_offline_env  # noqa: E402

apply_offline_env()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    os.environ.setdefault("YOLOWEB_RUN_DIR", str(run_dir))

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    params: dict = dict(config["params"])

    # sitecustomize 가 이미 등록했으면 install() 은 같은 콜백을 다시 붙이지 않는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "hooks"))
    import yoloweb_events

    yoloweb_events.install()

    try:
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        if SETTINGS.get("sync", True):
            SETTINGS.update({"sync": False})  # 단독망: 텔레메트리 전송 차단

        model = YOLO(params.pop("model"))
        model.train(
            data=config["data"],
            project=str(run_dir),
            name="train",
            exist_ok=True,
            **params,
        )
    except BaseException as exc:  # noqa: BLE001 - 어떤 실패든 이벤트로 남겨야 한다
        yoloweb_events.fail(exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
