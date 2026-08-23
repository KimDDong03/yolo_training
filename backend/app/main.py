"""FastAPI 앱 진입점.

단독망 + localhost 단독 사용이므로 인증을 두지 않는다. 대신 127.0.0.1 에만 바인드한다(run.py).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import datasets, jobs, presets, runs, system
from app.core import db
from app.core.config import FRONTEND_DIST, ensure_dirs
from app.services import dataset_prune, event_stream, run_manager
from app.services import jobs as job_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    db.connect()
    run_manager.recover()
    # 살아 있는 사이드잡을 다시 붙잡는다. 이걸 빼면 잡이 물고 있는 GPU 가 비어 보여
    # 스케줄러가 그 위에 학습을 띄운다 → 둘 다 OOM.
    job_service.recover()
    # 삭제 도중에 죽었으면 "파일은 없는데 목록에는 있는" 상태가 남는다 — 학습이 그걸로 죽는다.
    dataset_prune.recover()
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
app.include_router(jobs.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
    # 웹폰트는 Vite 의 public/ 을 거쳐 dist 최상단의 fonts/ 로 떨어진다 — assets/ 밖이라
    # 따로 걸어 준다. 없으면 아래 SPA 폴백이 index.html 을 대신 돌려주고 폰트가 조용히 안 뜬다.
    fonts_dir = FRONTEND_DIST / "fonts"
    if fonts_dir.is_dir():
        app.mount("/fonts", StaticFiles(directory=fonts_dir), name="fonts")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(index)
        return JSONResponse(
            {"detail": "프론트엔드가 빌드되지 않았습니다."}, status_code=404
        )
