"""저장소를 다른 경로로 clone 했을 때, 박혀 있는 절대경로를 새 경로로 고친다.

    python scripts/relocate.py

이 저장소에는 기존 학습 결과와 등록된 데이터셋이 함께 들어 있다. 그 메타데이터에는
만들어질 당시의 절대경로(예: C:\\Projects\\Yolo training\\storage\\...)가 그대로 들어 있어서,
다른 경로에 clone 하면 데이터셋을 찾지 못한다. 이 스크립트가 그걸 새 경로로 치환한다.

앱 코드는 건드리지 않는다. 고치는 대상은 아래 데이터 파일과 app.db 뿐이다.

storage/runs/*/train/args.yaml 과 train.log 는 ultralytics 가 학습 당시 남긴 기록이라
일부러 그대로 둔다. 화면 표시·이력 확인용이고 재실행에 쓰이지 않는다.

같은 경로에서 두 번 돌려도 안전하다 (옛 경로 == 새 경로면 아무것도 하지 않는다).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
DB_PATH = BASE_DIR / "app.db"

def _split_base(path_value: str) -> str | None:
    """'<옛 base>\\storage\\datasets\\...' 에서 '<옛 base>' 만 떼어낸다.

    rfind 로 뒤에서 찾는다. 레포 경로 자체에 'storage\\datasets\\' 가 들어 있어도
    (예: D:\\storage\\datasets\\yolo) 마지막 것이 우리가 원하는 구분점이다.
    """
    for sep in ("\\", "/"):
        needle = f"{sep}storage{sep}datasets{sep}"
        idx = path_value.rfind(needle)
        if idx > 0:
            return path_value[:idx]
    return None


def detect_old_base() -> str | None:
    """저장된 메타데이터에서 이 데이터가 만들어질 당시의 루트 경로를 역산한다."""
    for meta_path in sorted(STORAGE_DIR.glob("datasets/*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("yaml_path", "root"):
            value = meta.get(key)
            if isinstance(value, str):
                base = _split_base(value)
                if base:
                    return base

    for config_path in sorted(STORAGE_DIR.glob("runs/*/config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = config.get("data")
        if isinstance(value, str):
            base = _split_base(value)
            if base:
                return base

    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            row = conn.execute("SELECT yaml_path FROM datasets LIMIT 1").fetchone()
            conn.close()
        except sqlite3.Error:
            return None
        if row and row[0]:
            return _split_base(row[0])

    return None


def rewrite_file(path: Path, old: str, new: str) -> bool:
    """파일 안의 옛 경로를 새 경로로 바꾼다. 바꿨으면 True.

    JSON 파일에는 백슬래시가 이스케이프된 형태('C:\\\\Projects\\\\...')로 들어 있어서
    원시 형태와 이스케이프 형태를 모두 처리한다. 줄바꿈이 바뀌지 않도록 바이트로 다룬다.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return False

    text = raw.decode("utf-8")
    updated = text.replace(old.replace("\\", "\\\\"), new.replace("\\", "\\\\"))
    updated = updated.replace(old, new)

    if updated == text:
        return False
    path.write_bytes(updated.encode("utf-8"))
    return True


def rewrite_db(old: str, new: str) -> int:
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "UPDATE datasets SET root = replace(root, ?, ?), "
            "yaml_path = replace(yaml_path, ?, ?)",
            (old, new, old, new),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def main() -> int:
    new_base = str(BASE_DIR)

    old_base = detect_old_base()
    if old_base is None:
        print("고칠 경로를 찾지 못했다. 등록된 데이터셋도 학습 결과도 없는 상태로 보인다.")
        print("정상이다 — 데이터셋을 새로 등록하면 지금 경로로 기록된다.")
        return 0

    if old_base == new_base:
        print(f"경로가 이미 맞다: {new_base}")
        return 0

    print(f"옛 경로: {old_base}")
    print(f"새 경로: {new_base}")
    print()

    targets: list[Path] = []
    targets += sorted(STORAGE_DIR.glob("datasets/*/data.yaml"))
    targets += sorted(STORAGE_DIR.glob("datasets/*/train.txt"))
    targets += sorted(STORAGE_DIR.glob("datasets/*/val.txt"))
    targets += sorted(STORAGE_DIR.glob("datasets/*/meta.json"))
    targets += sorted(STORAGE_DIR.glob("runs/*/config.json"))

    changed = 0
    for target in targets:
        if rewrite_file(target, old_base, new_base):
            changed += 1
            print(f"  고침: {target.relative_to(BASE_DIR)}")

    rows = rewrite_db(old_base, new_base)

    print()
    print(f"파일 {changed}개, app.db 데이터셋 {rows}행을 새 경로로 고쳤다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
