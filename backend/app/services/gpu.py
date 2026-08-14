"""nvidia-smi 로 GPU 목록과 사용률을 읽는다.

여기서 GPU 개수를 하드코딩하지 않는 것이 요점이다. 3060 1장이든 B200 16장이든 같은 코드가 돈다.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time

_QUERY = "index,name,memory.total,memory.used,utilization.gpu"

# 화면이 2초마다 GPU 를 폴링하는데 이상 감지 감시자도 따로 물어본다.
# 캐시가 없으면 nvidia-smi 프로세스를 두 배로 띄우게 된다. 사용률은 1초 낡아도 무해하다.
CACHE_TTL_S = 1.0
_cache_lock = threading.Lock()
_cache: tuple[float, list[dict[str, object]]] | None = None


def list_gpus() -> list[dict[str, object]]:
    global _cache
    with _cache_lock:
        if _cache is not None and time.monotonic() - _cache[0] < CACHE_TTL_S:
            return _cache[1]
    gpus = _query_gpus()
    with _cache_lock:
        _cache = (time.monotonic(), gpus)
    return gpus


def _query_gpus() -> list[dict[str, object]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if out.returncode != 0:
        return []

    gpus: list[dict[str, object]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "utilization": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return gpus
