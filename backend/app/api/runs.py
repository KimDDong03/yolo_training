"""학습 실행 API + 실시간 스트림 WebSocket."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.core import db
from app.services import event_stream, gpu, run_manager

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _run_or_404(run_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        raise HTTPException(404, "실행을 찾을 수 없습니다.")
    return db.row_to_run(row)


@router.get("")
def list_runs() -> list[dict[str, Any]]:
    return [db.row_to_run(r) for r in db.query("SELECT * FROM runs ORDER BY created_at DESC")]


@router.get("/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = _run_or_404(run_id)
    dataset = db.query_one("SELECT * FROM datasets WHERE id = ?", (run["dataset_id"],))
    run["dataset"] = db.row_to_dataset(dataset) if dataset else None
    return run


@router.post("")
def create_run(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_row = db.query_one("SELECT * FROM datasets WHERE id = ?", (payload.get("dataset_id"),))
    if dataset_row is None:
        raise HTTPException(422, "데이터셋을 찾을 수 없습니다.")

    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise HTTPException(422, "params 는 객체여야 합니다.")
    try:
        epochs = int(params.get("epochs", 100))
    except (TypeError, ValueError):
        raise HTTPException(422, "epochs 가 숫자가 아닙니다.") from None
    if epochs < 1:
        raise HTTPException(422, "epochs 는 1 이상이어야 합니다.")

    devices = payload.get("devices")
    if devices is None:
        devices = [g["index"] for g in gpu.list_gpus()][:1]
    if not isinstance(devices, list) or any(not isinstance(d, int) for d in devices):
        raise HTTPException(422, "devices 는 GPU 번호의 배열이어야 합니다.")
    if len(devices) != len(set(devices)):
        # device="0,0" 을 넘기면 ultralytics 가 같은 GPU 를 두 장으로 알고 DDP 를 시도한다.
        raise HTTPException(422, "같은 GPU 번호를 두 번 지정할 수 없습니다.")
    known = {g["index"] for g in gpu.list_gpus()}
    unknown = [d for d in devices if d not in known]
    if unknown:
        raise HTTPException(422, f"존재하지 않는 GPU 번호입니다: {unknown}")

    dataset = db.row_to_dataset(dataset_row)
    name = str(payload.get("name") or f"{dataset['name']}")
    run = run_manager.create_run(name, dataset, params, devices)
    run_manager.schedule()
    return run


@router.post("/{run_id}/stop")
def stop_run(run_id: str, mode: str = "graceful") -> dict[str, Any]:
    _run_or_404(run_id)
    try:
        run_manager.stop_run(run_id, mode)
    except run_manager.RunError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _run_or_404(run_id)


@router.delete("/{run_id}")
def delete_run(run_id: str) -> dict[str, str]:
    run = _run_or_404(run_id)
    if run["status"] in {"running", "queued"}:
        raise HTTPException(409, "진행 중인 학습은 삭제할 수 없습니다. 먼저 정지하세요.")
    shutil.rmtree(run_manager.run_dir_for(run_id), ignore_errors=True)
    db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    return {"status": "deleted", "id": run_id}


@router.get("/{run_id}/events")
def run_events(run_id: str) -> dict[str, Any]:
    """WebSocket 없이도 전체 이벤트를 받아갈 수 있는 폴백."""
    _run_or_404(run_id)
    path = run_manager.run_dir_for(run_id) / "events.jsonl"
    events: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"events": events}


@router.get("/{run_id}/files/{path:path}")
def run_file(run_id: str, path: str) -> FileResponse:
    """run 디렉터리 안의 산출물(예측 이미지·플롯·가중치) 서빙."""
    _run_or_404(run_id)
    root = run_manager.run_dir_for(run_id).resolve()
    target = (root / path).resolve()
    if root not in target.parents:
        raise HTTPException(403, "실행 폴더 밖의 파일은 열 수 없습니다.")
    if not target.is_file():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    return FileResponse(target, filename=target.name if target.suffix == ".pt" else None)


@router.get("/{run_id}/artifacts")
def run_artifacts(run_id: str) -> dict[str, Any]:
    """종료 후 플롯·가중치 목록. end 이벤트를 놓쳤을 때도 화면이 채워지도록 직접 스캔한다."""
    _run_or_404(run_id)
    root = run_manager.run_dir_for(run_id)
    save_dir = root / "train"
    plots: list[str] = []
    weights: list[str] = []
    if save_dir.is_dir():
        for file in sorted(save_dir.glob("*.png")) + sorted(save_dir.glob("*.jpg")):
            if file.name.startswith(("val_batch", "train_batch")):
                continue
            plots.append(file.relative_to(root).as_posix())
        for file in sorted((save_dir / "weights").glob("*.pt")):
            weights.append(file.relative_to(root).as_posix())

    epochs: dict[str, list[str]] = {}
    epochs_dir = root / "epochs"
    if epochs_dir.is_dir():
        for folder in sorted(epochs_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            if folder.is_dir():
                epochs[folder.name] = [f.relative_to(root).as_posix() for f in sorted(folder.glob("*.jpg"))]
    return {"plots": plots, "weights": weights, "epochs": epochs}


@router.websocket("/{run_id}/ws")
async def run_ws(websocket: WebSocket, run_id: str) -> None:
    row = db.query_one("SELECT id FROM runs WHERE id = ?", (run_id,))
    if row is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    stream = event_stream.manager.get(run_id, run_manager.run_dir_for(run_id))
    snapshot, queue = stream.subscribe()
    try:
        await websocket.send_json(snapshot)
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
        pass
    finally:
        stream.unsubscribe(queue)
        await event_stream.manager.release(run_id)
