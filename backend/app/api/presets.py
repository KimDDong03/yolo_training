"""사용자 프리셋 — 자주 쓰는 파라미터 조합을 이름 붙여 저장한다."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core import db
from app.services import param_schema

router = APIRouter(prefix="/api/presets", tags=["presets"])


@router.get("")
def list_presets() -> dict[str, Any]:
    """내장 프리셋과 사용자 프리셋을 함께 내려준다."""
    user = [
        {
            "name": row["name"],
            "params": json.loads(row["params"]),
            "options": json.loads(row["options"]),
            "created_at": row["created_at"],
            "builtin": False,
        }
        for row in db.query("SELECT * FROM presets ORDER BY created_at DESC")
    ]
    builtin = [
        {
            "name": name,
            "params": patch,
            "options": {},
            "created_at": None,
            "builtin": True,
        }
        for name, patch in param_schema.PRESETS.items()
    ]
    return {"presets": builtin + user}


@router.post("")
def save_preset(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name", "")).strip()
    if not name:
        raise HTTPException(422, "프리셋 이름이 비어 있습니다.")
    if name in param_schema.PRESETS:
        raise HTTPException(409, f"내장 프리셋과 같은 이름은 쓸 수 없습니다: {name}")

    # 프리셋도 학습 요청과 같은 allowlist 를 통과해야 한다.
    # 여기서 막지 않으면 저장된 프리셋을 통해 임의 키가 train() 으로 흘러들어간다.
    try:
        params = param_schema.validate(payload.get("params"), "params")
        options = param_schema.validate(payload.get("options"), "options")
    except param_schema.ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc

    db.execute(
        "INSERT INTO presets (name, params, options, created_at) VALUES (?,?,?,?)"
        " ON CONFLICT(name) DO UPDATE SET params = excluded.params,"
        " options = excluded.options, created_at = excluded.created_at",
        (
            name,
            json.dumps(params, default=str),
            json.dumps(options, default=str),
            time.time(),
        ),
    )
    return {"name": name, "params": params, "options": options, "builtin": False}


@router.delete("/{name}")
def delete_preset(name: str) -> dict[str, str]:
    if name in param_schema.PRESETS:
        raise HTTPException(409, "내장 프리셋은 삭제할 수 없습니다.")
    cur = db.execute("DELETE FROM presets WHERE name = ?", (name,))
    if cur.rowcount == 0:
        raise HTTPException(404, "프리셋을 찾을 수 없습니다.")
    return {"status": "deleted", "name": name}
