"""파일시스템 정리 공통 동작."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

# Windows 에서는 방금 끝난 학습 프로세스나 백신이 핸들을 잠깐 더 쥐고 있는 일이 흔하다.
# 곧 풀릴 잠금 때문에 삭제를 실패로 보고하면 사용자가 멀쩡한 것을 두 번 지우게 된다.
_RETRY_DELAYS_S = (0, 0.2, 0.5, 1.0)


def remove_tree(path: Path) -> None:
    """폴더를 통째로 지운다. 끝내 못 지우면 OSError 를 낸다.

    ignore_errors=True 로 삼키면 안 되는 자리에 쓴다. 삼키면 파일이 그대로 남았는데도
    "지웠습니다" 라고 답하게 되고, DB 행만 사라져 디스크에는 참조 없는 가중치가 쌓인다.

    실패를 정리하는 경로(만들다 만 폴더 치우기)에서는 이 함수를 쓰지 마라.
    거기서는 원래 예외를 가리지 않는 것이 더 중요해서 ignore_errors=True 가 맞다.
    """
    if not path.exists():
        return

    last: OSError | None = None
    for delay in _RETRY_DELAYS_S:
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last = exc

    assert last is not None
    raise last
