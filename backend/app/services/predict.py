"""학습된 가중치로 임의 이미지에 추론한다.

자원 안전이 이 모듈의 핵심이다.

* **CPU 를 강제한다.** 학습이 도는 GPU 에 추론이 올라가면 학습이 OOM 으로 죽는다.
  요청으로 device 를 고를 수 있게 두면 언젠가 반드시 그 사고가 난다.
* **캐시 교체와 추론을 하나의 락으로 직렬화한다.** FastAPI 동기 엔드포인트는 스레드풀에서 병렬로
  실행되므로, 직렬화하지 않으면 동시에 두 모델이 로드돼 메모리가 두 배로 튄다.
* 캐시 키에 **mtime·size** 를 넣는다. 경로만으로 잡으면 같은 이름으로 덮어쓴 가중치를 옛것으로 계속 쓴다.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

MAX_UPLOAD_BYTES = 32 * 1024**2
MAX_PIXELS = 50_000_000  # 대략 8K 사진 한 장
KEEP_PREDICTIONS = 30

_lock = threading.Lock()
_cache_key: tuple[str, float, int] | None = None
_cache_model: Any = None


class PredictError(Exception):
    """사용자에게 그대로 보여줄 수 있는 추론 실패."""


def _load(weights: Path):
    """모델을 캐시에서 꺼내거나 새로 로드한다. 반드시 _lock 안에서 호출한다."""
    global _cache_key, _cache_model

    stat = weights.stat()
    key = (str(weights.resolve()), stat.st_mtime, stat.st_size)
    if _cache_key == key and _cache_model is not None:
        return _cache_model

    from ultralytics import YOLO

    _cache_model = YOLO(str(weights))
    _cache_key = key
    return _cache_model


def _prune(predictions_dir: Path) -> None:
    # 정렬 키가 stat() 을 부르므로, 다른 요청이 같은 파일을 지우는 중이면 FileNotFoundError 가 난다.
    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    files = sorted(predictions_dir.glob("*.jpg"), key=mtime, reverse=True)
    for stale in files[KEEP_PREDICTIONS:]:
        stale.unlink(missing_ok=True)


def run(
    run_dir: Path,
    weights: Path,
    image_bytes: bytes,
    conf: float,
    iou: float,
    imgsz: int,
) -> dict[str, Any]:
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise PredictError(
            f"이미지가 너무 큽니다 ({len(image_bytes) / 1024**2:.1f} MB)."
        )
    if not weights.is_file():
        raise PredictError(f"가중치를 찾을 수 없습니다: {weights.name}")

    import io

    import numpy as np
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise PredictError(
            "이미지를 열 수 없습니다. 손상되었거나 지원하지 않는 형식입니다."
        ) from exc

    if image.width * image.height > MAX_PIXELS:
        raise PredictError(
            f"이미지 해상도가 너무 큽니다 ({image.width}x{image.height})."
        )
    image = image.convert("RGB")

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    out = predictions_dir / f"{uuid.uuid4().hex}.jpg"

    started = time.time()
    with _lock:
        model = _load(weights)
        results = model.predict(
            source=np.asarray(image)[:, :, ::-1],  # ultralytics 는 BGR 배열을 받는다
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device="cpu",  # 학습 중인 GPU 와 경합하지 않기 위해 서버가 고정한다
            verbose=False,
        )
        # 저장과 정리도 같은 락 안에서 한다. 밖으로 빼면 동시 요청끼리 같은 파일을 지우다 충돌한다.
        result = results[0]
        annotated = result.plot()  # BGR ndarray
        Image.fromarray(annotated[:, :, ::-1]).save(out, quality=90)
        _prune(predictions_dir)
    elapsed_ms = (time.time() - started) * 1000

    names = getattr(result, "names", {}) or {}
    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for cls, score, xyxy in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            index = int(cls)
            detections.append(
                {
                    "cls": index,
                    "name": str(names.get(index, index)),
                    "conf": round(float(score), 4),
                    "xyxy": [round(float(v), 1) for v in xyxy],
                }
            )
    detections.sort(key=lambda d: d["conf"], reverse=True)

    return {
        "image": out.relative_to(run_dir).as_posix(),
        "detections": detections,
        "count": len(detections),
        "elapsed_ms": round(elapsed_ms),
        "weights": weights.name,
        "device": "cpu",
    }
