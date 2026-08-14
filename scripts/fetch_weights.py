"""사전학습 가중치를 bundle/weights/ 로 내려받는다.

    python scripts/fetch_weights.py

가중치는 저장소에 커밋하지 않는다. 이 앱의 모델 드롭다운은 bundle/weights/ 에
실제로 있는 파일만 "(번들)" 로 띄우므로, 이걸 한 번 돌려야 학습을 시작할 수 있다.

venv 활성화 전에도 돌아야 해서 표준 라이브러리만 쓴다.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "bundle" / "weights"

RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

# (파일명, 최소 크기) — 다운로드가 중간에 끊겨 생긴 조각 파일을 걸러내려고 크기를 본다.
WEIGHTS = [
    ("yolo11n.pt", 4 * 1024 * 1024),
    ("yolo26n.pt", 4 * 1024 * 1024),  # ultralytics AMP 체크가 이 파일을 요구한다
]


def download(name: str, dest: Path) -> None:
    url = f"{RELEASE}/{name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  받는 중: {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        tmp.write_bytes(response.read())
    tmp.replace(dest)


def main() -> int:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    for name, min_size in WEIGHTS:
        dest = WEIGHTS_DIR / name
        if dest.exists() and dest.stat().st_size >= min_size:
            print(f"[건너뜀] {name} — 이미 있음 ({dest.stat().st_size / 1024**2:.1f} MB)")
            continue
        try:
            download(name, dest)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"[실패] {name} — {exc}")
            failed.append(name)
            continue

        size = dest.stat().st_size
        if size < min_size:
            print(f"[실패] {name} — 파일이 너무 작다 ({size} bytes). 지운다.")
            dest.unlink()
            failed.append(name)
        else:
            print(f"[완료] {name} — {size / 1024**2:.1f} MB")

    if failed:
        print()
        print("아래 파일을 직접 받아 bundle/weights/ 에 넣어라:")
        for name in failed:
            print(f"  {RELEASE}/{name}")
        return 1

    print()
    print(f"가중치 준비 완료: {WEIGHTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
