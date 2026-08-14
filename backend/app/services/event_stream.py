"""events.jsonl / train.log 를 tail 해서 WebSocket 구독자에게 팬아웃한다.

설계상 중요한 두 가지:
  1. 파일이 단일 진실 원천이다. 백엔드 메모리는 파일을 다시 읽어 언제든 복원되는 캐시일 뿐이다.
  2. 리더는 run 당 하나다. 탭을 20개 열어도 파일 핸들은 1개.

부분 기록된 줄(워커가 append 하는 도중에 읽은 경우)은 '\n' 이 올 때까지 버퍼에 남기고
읽기 위치를 전진시키지 않는다. 이게 이 방식에서 유일하게 까다로운 지점이다.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

POLL_INTERVAL = 0.25
LOG_TAIL_LINES = 2000
# 차트 복원에 필요한 이벤트만 메모리에 남긴다. batch 는 진행률이라 최신 하나면 충분하다.
# warning 이 빠지면 이상 감지 배지가 라이브에서만 보이고 새로고침하면 사라진다.
KEEP_KINDS = {"start", "epoch", "final_val", "artifact", "checkpoint", "end", "warning"}


def json_safe(obj: Any) -> Any:
    """파싱된 이벤트에서 NaN/Inf 를 None 으로 바꾼다.

    파이썬 json 은 NaN 리터럴을 읽고 또 그대로 쓰지만, 브라우저 JSON.parse 는 거기서
    SyntaxError 로 죽는다. 워커는 이제 non-finite 를 쓰지 않지만
    (hooks/yoloweb_events.py 의 _num), 그 수정 이전에 만들어진 events.jsonl 이
    디스크에 남아 있다. 과거 run 을 열었다고 스트림이 멎으면 안 된다.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_safe(value) for value in obj]
    return obj


class _Tailer:
    """파일을 뒤에서부터 이어 읽으며 완전한 줄만 돌려준다."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self._buffer = b""
        self._identity: tuple[int, int] | None = None

    def read_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            stat = self.path.stat()
        except OSError:
            return []
        size = stat.st_size

        # 파일이 통째로 교체되면 크기만으로는 알 수 없다. 파일 식별자가 바뀌면 처음부터 다시 읽는다.
        identity = (stat.st_dev, stat.st_ino)
        if self._identity is None:
            self._identity = identity
        elif identity != self._identity:
            self._identity = identity
            self.offset = 0
            self._buffer = b""

        if size < self.offset:  # 잘린 경우
            self.offset = 0
            self._buffer = b""
        if size == self.offset:
            return []

        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read(size - self.offset)
        except OSError:
            return []

        self.offset += len(chunk)
        self._buffer += chunk
        *complete, self._buffer = self._buffer.split(b"\n")
        return [
            line.decode("utf-8", errors="replace").rstrip("\r")
            for line in complete
            if line.strip()
        ]


class RunStream:
    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.events = _Tailer(run_dir / "events.jsonl")
        self.logs = _Tailer(run_dir / "train.log")
        self.history: list[dict[str, Any]] = []
        self.last_batch: dict[str, Any] | None = None
        self.log_tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self.finished = False
        self._subscribers: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------- 수명주기
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    # -------------------------------------------------------------- 구독
    def subscribe(self) -> tuple[dict[str, Any], asyncio.Queue]:
        """스냅샷과 라이브 큐를 원자적으로 함께 넘긴다.

        큐를 먼저 등록한 뒤 스냅샷을 뜨므로, 그 사이에 도착한 이벤트는 큐에 들어가고
        스냅샷에도 들어갈 수 있다. 중복은 프론트가 (kind, epoch) 로 흡수한다.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(queue)
        snapshot = {
            "type": "snapshot",
            "run_id": self.run_id,
            "events": list(self.history),
            "batch": self.last_batch,
            "logs": list(self.log_tail),
            "finished": self.finished,
        }
        return snapshot, queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def _publish(self, message: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # 느린 구독자 때문에 리더가 막히면 안 된다. 그 구독자는 새로고침하면 복구된다.
                self._subscribers.discard(queue)

    # -------------------------------------------------------------- 리더
    def pump(self) -> None:
        """파일에서 새 내용을 읽어 메모리 캐시와 구독자에게 반영한다 (동기)."""
        for line in self.events.read_lines():
            try:
                obj = json_safe(json.loads(line))
            except json.JSONDecodeError:
                continue
            kind = obj.get("t")
            if kind == "batch":
                self.last_batch = obj
            elif kind in KEEP_KINDS:
                self.history.append(obj)
                if kind == "end":
                    self.finished = True
            self._publish({"type": "event", "event": obj})

        new_logs = self.logs.read_lines()
        if new_logs:
            self.log_tail.extend(new_logs)
            self._publish({"type": "log", "lines": new_logs})

    async def _loop(self) -> None:
        while True:
            self.pump()
            await asyncio.sleep(POLL_INTERVAL)


class StreamManager:
    def __init__(self) -> None:
        self._streams: dict[str, RunStream] = {}

    def get(self, run_id: str, run_dir: Path) -> RunStream:
        stream = self._streams.get(run_id)
        if stream is None:
            stream = RunStream(run_id, run_dir)
            stream.pump()  # 과거 이벤트를 먼저 전부 읽어들인다 (새로고침 복원의 근거)
            self._streams[run_id] = stream
        stream.start()
        return stream

    async def release(self, run_id: str) -> None:
        """구독자가 없으면 tail 태스크를 정리한다.

        학습이 아직 끝나지 않았어도 멈춘다. 파일이 단일 진실 원천이라 다시 접속하면
        처음부터 읽어 복원되기 때문이다. 여기서 붙잡고 있으면 아무도 안 보는 run 마다
        0.25초 폴링 태스크가 영원히 남는다.
        """
        stream = self._streams.get(run_id)
        if stream and stream.subscriber_count == 0:
            await stream.stop()
            self._streams.pop(run_id, None)

    async def shutdown(self) -> None:
        for stream in list(self._streams.values()):
            await stream.stop()
        self._streams.clear()


manager = StreamManager()
