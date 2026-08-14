"""SQLite 메타 저장소.

진행 상황·지표는 여기 저장하지 않는다. 그건 events.jsonl 이 단일 원천이다.
여기 있는 건 "무엇이 존재하고 지금 어떤 상태인가" 뿐이다.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from app.core.config import DB_PATH

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    source      TEXT NOT NULL,          -- 'zip' | 'path'
    origin      TEXT,                   -- 업로드 파일명 또는 원본 폴더 경로
    root        TEXT NOT NULL,          -- 이미지가 실제로 있는 루트
    yaml_path   TEXT NOT NULL,          -- 학습에 넘길 data.yaml
    classes     TEXT NOT NULL,          -- JSON 배열
    report      TEXT NOT NULL,          -- JSON 검수 리포트
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    dataset_id   TEXT NOT NULL,
    status       TEXT NOT NULL,         -- queued|running|completed|stopped|failed
    params       TEXT NOT NULL,         -- JSON
    devices      TEXT NOT NULL,         -- JSON 배열 (GPU index, 빈 배열이면 CPU)
    pid          INTEGER,
    error        TEXT,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL
);
"""


def connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.executescript(SCHEMA)
            _conn.commit()
    return _conn


def execute(sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
    conn = connect()
    with _lock:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def query(sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    conn = connect()
    with _lock:
        return conn.execute(sql, params).fetchall()


def query_one(sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def row_to_dataset(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "source": row["source"],
        "origin": row["origin"],
        "root": row["root"],
        "yaml_path": row["yaml_path"],
        "classes": json.loads(row["classes"]),
        "report": json.loads(row["report"]),
        "created_at": row["created_at"],
    }


def row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "dataset_id": row["dataset_id"],
        "status": row["status"],
        "params": json.loads(row["params"]),
        "devices": json.loads(row["devices"]),
        "pid": row["pid"],
        "error": row["error"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }
