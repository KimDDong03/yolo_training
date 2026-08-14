"""테스트가 backend 패키지를 import 할 수 있게 경로를 잡고, 임시 저장소를 만들어 준다.

여기서 임시 경로를 쓰는 이유: config 의 경로는 모듈 상수라 사용자의 실제 app.db 와
storage/ 를 그대로 가리킨다. 테스트가 그걸 건드리면 안 된다.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def isolate_storage() -> Path:
    """DB 와 runs 폴더를 임시 경로로 돌린다. 임시 루트를 돌려준다."""
    root = Path(tempfile.mkdtemp(prefix="yoloweb_test_"))

    from app.core import config, db
    from app.services import run_manager

    config.DB_PATH = root / "app.db"
    config.RUNS_DIR = root / "runs"
    config.DATASETS_DIR = root / "datasets"
    config.STORAGE_DIR = root
    db.DB_PATH = config.DB_PATH
    db._conn = None  # 이전 테스트가 열어 둔 연결을 버린다
    run_manager.RUNS_DIR = config.RUNS_DIR

    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return root
