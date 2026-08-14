"""ultralytics 콜백에서 events.jsonl 을 쓰는 실제 구현.

이 모듈은 sitecustomize.py 가 불러 ultralytics 의 전역 default_callbacks 에 직접 등록한다.
model.add_callback() 을 쓰지 않는 이유: 멀티 GPU(DDP)일 때 ultralytics 가 임시 .py 파일로
트레이너를 새로 조립해 자식 프로세스에서 돌리기 때문에(ultralytics/utils/dist.py) 모델에 붙인
콜백이 자식 rank 로 전달되지 않는다. 전역 등록 + PYTHONPATH 상속이면 어느 프로세스든 붙는다.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

BATCH_INTERVAL = 0.5


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", -1))
    except ValueError:
        return -1


def _scalar(value: Any) -> Any:
    """torch/numpy 스칼라를 파이썬 값으로 바꾼다. non-finite 판정은 하지 않는다."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    # 학습 중 loss 는 requires_grad=True 인 텐서라 그냥 float() 하면 경고가 뜬다.
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value.item())
        except Exception:
            return None


def _is_nonfinite(value: Any) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def _num(value: Any) -> Any:
    """JSON 으로 쓸 수 있는 값으로 바꾼다. NaN/Inf 는 None 으로 떨군다.

    json.dumps 는 기본값(allow_nan=True)으로 NaN 을 그대로 `NaN` 리터럴로 쓴다.
    파이썬 json.loads 는 그걸 읽지만 브라우저 JSON.parse 는 SyntaxError 로 죽어
    (api/runs.py 의 send_json → useRunStream 의 onmessage) 스트림 전체가 멎는다.
    하필 loss 가 발산하는 그 순간에 화면이 통째로 멈추는 것이라, 값을 버리더라도
    줄은 파싱 가능해야 한다. 버려진 사실은 호출부가 loss_nan / nonfinite 로 남긴다.
    """
    value = _scalar(value)
    return None if _is_nonfinite(value) else value


def _nonfinite_keys(values: dict[str, Any]) -> list[str]:
    """_num 을 거치면 사라질(NaN/Inf 였던) 키를 미리 골라낸다."""
    return sorted(key for key, value in values.items() if _is_nonfinite(_scalar(value)))


def _cuda_mem_gb(trainer: Any) -> float | None:
    """이 run 이 지금까지 잡아 본 VRAM 최대치(GB). CPU 학습이면 None.

    학습 시작 전에 "얼마나 걸리고 VRAM 이 얼마나 드는지" 를 예측하려면 실측 표본이 있어야
    하는데, 시간(epoch_time_s)과 달리 VRAM 은 어디에도 남지 않는다. 계측이 예측보다 먼저다.
    reserved 를 쓰는 이유는 실제로 다른 프로세스가 쓸 수 없는 양이 그쪽이기 때문이다.

    판정 기준은 torch.cuda.is_available() 이 아니라 트레이너가 실제로 쓰는 device 다.
    GPU 가 꽂힌 PC 에서 CPU 로 학습하면 available 은 True 라, 그걸로 재면 0.0 이 실측값처럼
    기록되어 나중에 VRAM 추정을 오염시킨다.
    """
    device = getattr(trainer, "device", None)
    # torch.device 면 .type, 문자열이면 "cuda:0" 의 앞부분. 둘 다 받아 둔다 —
    # 여기서 못 알아보면 GPU run 의 VRAM 이 조용히 안 남고, 그건 나중에야 드러난다.
    kind = getattr(device, "type", None) or str(device or "").split(":")[0]
    if kind != "cuda":
        return None
    try:
        import torch

        return round(torch.cuda.max_memory_reserved(device) / 1e9, 3)
    except Exception:
        return None


def _summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    """ultralytics 의 원본 키를 그대로 두되, 차트가 바로 쓸 표준 이름을 함께 만든다.

    키 이름을 추측해서 하나만 싣지 않는다. 버전이 올라 키가 바뀌면 차트가 조용히 비기 때문이다.
    """
    summary: dict[str, Any] = {}
    for key, value in metrics.items():
        low = key.lower()
        if "map50-95" in low:
            summary.setdefault("mAP50-95", _num(value))
        elif "map50" in low:
            summary.setdefault("mAP50", _num(value))
        elif "precision" in low:
            summary.setdefault("precision", _num(value))
        elif "recall" in low:
            summary.setdefault("recall", _num(value))
    return summary


class EventWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "events.jsonl"
        self.stop_flag = run_dir / "stop.flag"
        self.epochs_dir = run_dir / "epochs"
        self.last_batch_at = 0.0
        self.train_start = time.time()
        self.ended = False
        # 배치 인덱스는 트레이너 속성으로 노출되지 않아(루프 지역변수) 콜백 호출 횟수로 센다.
        self.batch_epoch = -1
        self.batch_count = 0
        self.last_epoch_emitted = -1

    def write(self, payload: dict[str, Any]) -> None:
        if _rank() not in (-1, 0):
            return
        payload.setdefault("ts", time.time())
        line = json.dumps(payload, ensure_ascii=False, default=str)
        # 한 줄을 한 번의 write 로 내보내야 리더가 잘린 JSON 을 보지 않는다.
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def stop_requested(self) -> bool:
        return self.stop_flag.exists()


_writer: EventWriter | None = None


def _w() -> EventWriter | None:
    return _writer


# ------------------------------------------------------------------ 콜백들


def on_train_start(trainer) -> None:
    writer = _w()
    if not writer:
        return
    writer.train_start = time.time()
    names = (
        getattr(getattr(trainer, "data", None), "get", lambda *_: None)("names") or {}
    )
    writer.write(
        {
            "t": "start",
            "total_epochs": int(getattr(trainer, "epochs", 0) or 0),
            "model": str(getattr(trainer.args, "model", "")),
            "imgsz": getattr(trainer.args, "imgsz", None),
            "batch": getattr(trainer.args, "batch", None),
            "device": str(getattr(trainer, "device", "")),
            "save_dir": str(getattr(trainer, "save_dir", "")),
            "classes": [
                str(v) for v in (names.values() if hasattr(names, "values") else names)
            ],
        }
    )


def on_train_batch_end(trainer) -> None:
    writer = _w()
    if not writer:
        return
    epoch = int(getattr(trainer, "epoch", 0)) + 1
    if epoch != writer.batch_epoch:
        writer.batch_epoch = epoch
        writer.batch_count = 0
    writer.batch_count += 1

    now = time.time()
    if now - writer.last_batch_at >= BATCH_INTERVAL:
        writer.last_batch_at = now
        loader = getattr(trainer, "train_loader", None)
        try:
            total = len(loader) if loader is not None else None
        except TypeError:
            total = None
        loss = _scalar(getattr(trainer, "loss", None))
        payload = {
            "t": "batch",
            "epoch": epoch,
            "i": writer.batch_count,
            "n": total,
            "loss": None if _is_nonfinite(loss) else loss,
        }
        if _is_nonfinite(loss):
            payload["loss_nan"] = True
        writer.write(payload)

    # 단일 GPU 에서만 배치 경계에서 즉시 멈춘다.
    # DDP 에서 rank 0 만 break 하면 다른 rank 가 collective 에서 멈춰 데드락이 된다
    # (ultralytics/engine/trainer.py:500-502 에는 broadcast 가 없다).
    if _rank() == -1 and writer.stop_requested():
        trainer.stop = True


def on_fit_epoch_end(trainer) -> None:
    writer = _w()
    if not writer:
        return

    metrics: dict[str, Any] = {}
    try:
        metrics.update(trainer.label_loss_items(trainer.tloss))
    except Exception:
        pass
    for key, value in (getattr(trainer, "metrics", None) or {}).items():
        metrics[key] = value
    nonfinite = _nonfinite_keys(metrics)
    metrics = {k: _num(v) for k, v in metrics.items()}

    epoch = int(getattr(trainer, "epoch", 0)) + 1
    total = int(getattr(trainer, "epochs", 0) or 0)
    epoch_time = _num(getattr(trainer, "epoch_time", None)) or 0.0

    # 학습 종료 후 final_eval() 이 on_fit_epoch_end 를 한 번 더 부른다(trainer.py:845).
    # 같은 에폭 번호로 두 번 찍히면 차트에 점이 겹치므로 별도 종류로 남긴다.
    kind = "epoch" if epoch != writer.last_epoch_emitted else "final_val"
    writer.last_epoch_emitted = epoch
    payload = {
        "t": kind,
        "epoch": epoch,
        "total_epochs": total,
        "metrics": metrics,
        "summary": _summarize(metrics),
        "lr": {k: _num(v) for k, v in (getattr(trainer, "lr", None) or {}).items()},
        "fitness": _num(getattr(trainer, "fitness", None)),
        "best_fitness": _num(getattr(trainer, "best_fitness", None)),
        "epoch_time_s": epoch_time,
        "eta_s": epoch_time * max(total - epoch, 0),
        "mem_gb": _cuda_mem_gb(trainer),
    }
    if nonfinite:
        payload["nonfinite"] = nonfinite
    writer.write(payload)

    _copy_epoch_images(writer, trainer, epoch)

    # 안전 정지: 이 시점에 stop 을 세우면 trainer.py:555-558 의 broadcast 를 타고
    # 모든 rank 에 전파되므로 DDP 에서도 안전하다.
    if writer.stop_requested():
        trainer.stop = True


def on_val_start(validator) -> None:
    """에폭마다 검증 예측 이미지를 그리게 강제한다.

    ultralytics 는 기본적으로 마지막 에폭에서만 val_batch*_pred.jpg 를 만든다
    (engine/validator.py:163 의 `self.args.plots &= ... trainer.epoch == trainer.epochs - 1`).
    그대로 두면 "에폭별 예측 변화를 슬라이더로 훑는" 기능 자체가 성립하지 않는다.
    이 콜백은 그 감쇠 연산(:163) 뒤, 실제 플롯 지점(:242) 앞인 on_val_start(:211) 에서 돈다.
    """
    writer = _w()
    if not writer or not getattr(validator, "training", False):
        return
    try:
        validator.args.plots = True
    except Exception:
        pass


def _copy_epoch_images(writer: EventWriter, trainer, epoch: int) -> None:
    save_dir = Path(getattr(trainer, "save_dir", "") or "")
    if not save_dir.is_dir():
        return
    target = writer.epochs_dir / str(epoch)
    files: list[str] = []
    for pattern in ("val_batch*_pred.jpg", "val_batch*_labels.jpg"):
        for src in sorted(save_dir.glob(pattern)):
            target.mkdir(parents=True, exist_ok=True)
            dst = target / src.name
            try:
                shutil.copyfile(src, dst)
            except OSError:
                continue
            files.append(dst.relative_to(writer.run_dir).as_posix())
    if files:
        writer.write(
            {"t": "artifact", "kind": "val_pred", "epoch": epoch, "files": files}
        )


def on_model_save(trainer) -> None:
    writer = _w()
    if not writer:
        return
    writer.write(
        {
            "t": "checkpoint",
            "epoch": int(getattr(trainer, "epoch", 0)) + 1,
            "best_fitness": _num(getattr(trainer, "best_fitness", None)),
        }
    )


def on_train_end(trainer) -> None:
    writer = _w()
    if not writer or writer.ended:
        return
    writer.ended = True

    save_dir = Path(getattr(trainer, "save_dir", "") or "")
    plots: list[str] = []
    weights: list[str] = []
    if save_dir.is_dir():
        for path in sorted(save_dir.glob("*.png")) + sorted(save_dir.glob("*.jpg")):
            if path.name.startswith("val_batch") or path.name.startswith("train_batch"):
                continue
            try:
                plots.append(path.relative_to(writer.run_dir).as_posix())
            except ValueError:
                continue
        for path in sorted((save_dir / "weights").glob("*.pt")):
            try:
                weights.append(path.relative_to(writer.run_dir).as_posix())
            except ValueError:
                continue

    metrics = {k: _num(v) for k, v in (getattr(trainer, "metrics", None) or {}).items()}
    writer.write(
        {
            "t": "end",
            "status": "stopped" if writer.stop_requested() else "completed",
            "epochs_done": int(getattr(trainer, "epoch", 0)) + 1,
            "best_metrics": metrics,
            "summary": _summarize(metrics),
            "plots": plots,
            "weights": weights,
            "elapsed_s": time.time() - writer.train_start,
        }
    )


def fail(error: BaseException) -> None:
    """워커가 예외로 죽을 때 마지막 이벤트를 남긴다."""
    writer = _w()
    if not writer or writer.ended:
        return
    writer.ended = True
    writer.write(
        {
            "t": "end",
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    )


def _set_in_memory(settings, key: str, value) -> None:
    """ultralytics 설정을 이 프로세스 안에서만 바꾼다.

    SETTINGS 는 JSONDict 라서 `SETTINGS[k] = v` 나 `.update()` 가 곧바로
    %APPDATA%\\Ultralytics\\settings.json 에 쓴다. 그러면 동시에 도는 run 들이 서로 값을 덮어쓰고,
    사용자의 전역 설정까지 바뀐다. dict.__setitem__ 으로 우회해 메모리에만 반영한다.
    """
    try:
        dict.__setitem__(settings, key, value)
    except Exception:
        pass


def install() -> bool:
    """전역 default_callbacks 에 등록한다. 성공하면 True."""
    global _writer
    run_dir = os.environ.get("YOLOWEB_RUN_DIR")
    if not run_dir:
        return False
    _writer = EventWriter(Path(run_dir))

    from ultralytics.utils import SETTINGS
    from ultralytics.utils.callbacks import base

    # 단독망: 텔레메트리 전송 차단. 설정 파일은 건드리지 않는다.
    _set_in_memory(SETTINGS, "sync", False)
    # TensorBoard 는 UI 옵션이다. 통합 콜백이 로드되기 전인 지금 반영해야 하고,
    # sitecustomize 를 통해 DDP 자식 프로세스에서도 같은 코드가 돈다.
    _set_in_memory(
        SETTINGS, "tensorboard", os.environ.get("YOLOWEB_TENSORBOARD") == "1"
    )

    hooks = {
        "on_train_start": on_train_start,
        "on_train_batch_end": on_train_batch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_start": on_val_start,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
    }
    for name, fn in hooks.items():
        callbacks = base.default_callbacks.setdefault(name, [])
        if fn not in callbacks:
            callbacks.append(fn)
    return True
