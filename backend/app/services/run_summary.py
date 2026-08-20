"""실행 목록에 얹을 한 줄 요약.

사이드바가 목록에서 바로 "37/100" 과 "mAP50-95 0.612" 를 보여준다. 그 값은 DB 에 없고
events.jsonl 에만 있다. 그렇다고 실행마다 /events 를 따로 부르면 목록 요청이 N+1 이 된다.

목록은 2초마다 폴링된다(frontend App.tsx). 매번 실행 57개의 events.jsonl 을 처음부터 읽으면
목록 요청 하나가 수십 MB 를 읽는다. 그래서 (mtime, size) 를 키로 캐시한다 —
끝난 실행은 파일이 변하지 않아 영구히 맞고, 실제로 다시 읽는 것은 도는 실행(보통 1개)뿐이다.

한 번 훑으면서 최고 mAP 와 마지막 에폭을 같이 뽑는다. 두 값을 따로 구하면 파일을 두 번 읽는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EMPTY: dict[str, Any] = {"epoch": None, "total_epochs": None, "best_map": None}

# path -> ((mtime_ns, size), 요약). 실행이 지워져도 항목이 남지만 한 건이 수십 바이트라
# 재기동 전까지 두어도 문제가 없다. 그래도 무한히 늘지는 않게 상한을 둔다.
_CACHE: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
_CACHE_MAX = 512


def summarize(run_dir: Path) -> dict[str, Any]:
    """events.jsonl 에서 진행률과 최고 mAP50-95 를 뽑는다. 못 읽으면 전부 None 이다.

    목록 응답이 이것 때문에 실패하면 안 되므로 어떤 예외도 밖으로 내보내지 않는다.
    """
    path = run_dir / "events.jsonl"
    try:
        stat = path.stat()
    except OSError:
        return dict(EMPTY)

    key = (stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == key:
        return dict(cached[1])

    result = _scan(path)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[path] = (key, result)
    return dict(result)


def _scan(path: Path) -> dict[str, Any]:
    epoch: int | None = None
    total: int | None = None
    best: float | None = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                # 배치 이벤트가 수만 줄 쌓인다. json.loads 를 부르기 전에 걸러 낸다.
                if '"epoch"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 문법이 맞아도 객체가 아닐 수 있다("epoch" 같은 문자열 한 줄). dict 가 아니면
                # .get 이 AttributeError 로 터지고 그 예외가 목록 응답 전체를 500 으로 만든다.
                if not isinstance(obj, dict) or obj.get("t") != "epoch":
                    continue
                value = obj.get("epoch")
                if isinstance(value, int):
                    epoch = value
                value = obj.get("total_epochs")
                if isinstance(value, int):
                    total = value
                summary = obj.get("summary")
                value = summary.get("mAP50-95") if isinstance(summary, dict) else None
                if isinstance(value, (int, float)) and (best is None or value > best):
                    best = float(value)
    except OSError:
        return dict(EMPTY)
    return {"epoch": epoch, "total_epochs": total, "best_map": best}
