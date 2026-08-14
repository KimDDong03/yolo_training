"""nvidia-smi 로 GPU 목록과 사용률을 읽는다.

여기서 GPU 개수를 하드코딩하지 않는 것이 요점이다. 3060 1장이든 B200 16장이든 같은 코드가 돈다.
"""

from __future__ import annotations

import shutil
import subprocess

_QUERY = "index,name,memory.total,memory.used,utilization.gpu"


def list_gpus() -> list[dict[str, object]]:
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
