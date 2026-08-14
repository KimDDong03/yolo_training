"""FastAPI 앱 진입점.

단독망 + localhost 단독 사용이므로 인증을 두지 않는다. 대신 127.0.0.1 에만 바인드한다(run.py).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import datasets, presets, runs, system
from app.core import db
from app.core.config import FRONTEND_DIST, ensure_dirs
from app.services import event_stream, run_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    db.connect()
    run_manager.recover()
    task = asyncio.create_task(run_manager.scheduler_loop())
    try:
        yield
    finally:
        task.cancel()
        await event_stream.manager.shutdown()


app = FastAPI(title="YOLO 학습 콘솔", lifespan=lifespan)
app.include_router(system.router)
app.include_router(presets.router)
app.include_router(datasets.router)
app.include_router(runs.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(index)
        return JSONResponse(
            {"detail": "프론트엔드가 빌드되지 않았습니다."}, status_code=404
        )
