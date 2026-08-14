"""내보내기 워커 — ONNX / TorchScript / TensorRT 변환을 독립 프로세스로 수행한다.

TensorRT 빌드는 수 분이 걸리고 VRAM 을 크게 쓴다. 웹서버 안에서 돌리면 서버가 멈춘 것처럼 보인다.
학습과 같은 방식으로 프로세스를 분리하고 진행 상황을 파일에 남긴다.

사용법: python export_worker.py --run-dir <dir> --format onnx --weights train/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import apply_offline_env  # noqa: E402

apply_offline_env()


def write(path: Path, payload: dict) -> None:
    payload.setdefault("ts", time.time())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    # 잡 관리로 넘어오면서 진행 상황이 소유자 폴더 아래로 모인다.
    # 인자가 없으면 예전 경로에 쓴다 — 이 워커를 직접 돌리던 방식을 깨지 않는다.
    parser.add_argument("--events", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    events = Path(args.events).resolve() if args.events else run_dir / "export.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    weights = (run_dir / args.weights).resolve()

    write(
        events,
        {
            "t": "start",
            "format": args.format,
            "weights": args.weights,
            "imgsz": args.imgsz,
            "device": args.device,
            "half": args.half,
            "dynamic": args.dynamic,
        },
    )

    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        kwargs: dict = {
            "format": args.format,
            "imgsz": args.imgsz,
            "device": args.device,
        }
        if args.half:
            kwargs["half"] = True
        if args.dynamic:
            kwargs["dynamic"] = True

        output = model.export(**kwargs)
        path = Path(str(output))
        rel = (
            path.relative_to(run_dir).as_posix()
            if run_dir in path.parents
            else str(path)
        )
        write(
            events,
            {
                "t": "end",
                "status": "completed",
                "file": rel,
                "size_mb": round(path.stat().st_size / 1024**2, 1)
                if path.is_file()
                else None,
            },
        )
    except BaseException as exc:  # noqa: BLE001 - 실패 원인을 그대로 화면에 보여준다
        write(
            events,
            {
                "t": "end",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
