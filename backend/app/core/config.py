"""경로·상한값·오프라인 환경 설정.

이 모듈은 ultralytics 를 import 하기 전에 먼저 import 되어야 한다.
YOLO_OFFLINE 을 세워두지 않으면 ultralytics 가 import 시점에 DNS 를 조회해
단독망에서 기동이 수 초씩 지연된다 (ultralytics/utils/__init__.py 의 ONLINE = is_online()).
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = BASE_DIR / "backend"
HOOKS_DIR = BACKEND_DIR / "hooks"
STORAGE_DIR = BASE_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
RUNS_DIR = STORAGE_DIR / "runs"
UPLOADS_DIR = STORAGE_DIR / "uploads"
BUNDLE_DIR = BASE_DIR / "bundle"
WEIGHTS_DIR = BUNDLE_DIR / "weights"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
DB_PATH = BASE_DIR / "app.db"

HOST = "127.0.0.1"
PORT = 8000

# zip 해제 상한 (zip 폭탄 방어)
MAX_ZIP_BYTES = 8 * 1024**3
MAX_UNCOMPRESSED_BYTES = 32 * 1024**3
MAX_ENTRIES = 500_000
ALLOWED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 학습 워커가 events.jsonl 에 배치 진행률을 쓰는 최소 간격(초)
BATCH_EVENT_INTERVAL = 0.5


def offline_env() -> dict[str, str]:
    """워커 서브프로세스에 물려줄 오프라인 관련 환경변수."""
    return {
        "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false",
    }


def apply_offline_env() -> None:
    """현재 프로세스에 오프라인 환경변수를 적용한다 (ultralytics import 전에 호출)."""
    for key, value in offline_env().items():
        os.environ.setdefault(key, value)


def ensure_dirs() -> None:
    for path in (STORAGE_DIR, DATASETS_DIR, RUNS_DIR, UPLOADS_DIR, WEIGHTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


apply_offline_env()
