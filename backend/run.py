"""개발/운영 공통 실행 진입점.

    python backend/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import HOST, PORT, apply_offline_env, ensure_dirs  # noqa: E402

apply_offline_env()
ensure_dirs()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
