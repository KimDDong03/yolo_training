"""워커 프로세스의 생존·신원 확인과 종료.

학습 워커와 사이드잡(내보내기·분석 등)이 같은 문제를 공유한다. 백엔드가 재시작되어도
워커는 살아 있고(Windows 는 부모가 죽어도 자식이 함께 죽지 않는다), 그걸 PID 로 다시
붙잡아야 한다. 그런데 PID 는 재사용된다 — 그 사이 워커가 죽고 같은 번호가 다른
프로세스에 할당됐다면, 강제 종료할 때 무관한 프로세스 트리를 죽이게 된다.

그래서 "살아 있는가" 와 "그게 정말 우리가 띄운 그것인가" 를 나눠서 판단한다.
"""

from __future__ import annotations

import os
import subprocess

# PID 재사용 방어: 프로세스 생성 시각이 기준 시각과 이만큼 이상 벌어지면 다른 프로세스로 본다.
PID_IDENTITY_WINDOW_S = 300

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


def _win_process_info(pid: int) -> tuple[bool, float | None]:
    """(살아있는가, 생성시각 unix초). Windows 전용."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False, None
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False, None
        if code.value != _STILL_ACTIVE:
            return False, None
        created, exited, kernel_t, user_t = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel_t),
            ctypes.byref(user_t),
        ):
            return True, None
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        # FILETIME: 1601-01-01 기준 100ns 단위
        return True, ticks / 1e7 - 11644473600.0
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """PID 생존 확인.

    Windows 에서 os.kill(pid, 0) 은 실제로 프로세스를 죽이므로 절대 쓰면 안 된다.
    """
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
    return _win_process_info(pid)[0]


def is_our_worker(pid: int | None, started_at: float | None) -> bool:
    """이 PID 가 정말 우리가 띄운 그 프로세스인지 확인한다.

    생성 시각이 기록해 둔 시작 시각 근처인지까지 본다. 이 검증 없이 kill 하면
    재사용된 PID 를 타고 무관한 프로세스를 죽일 수 있다.
    """
    if not pid or pid <= 0:
        return False
    if os.name != "nt":
        return pid_alive(int(pid))
    alive, created = _win_process_info(int(pid))
    if not alive:
        return False
    if started_at is None or created is None:
        return False  # 확인할 수 없으면 우리 것으로 간주하지 않는다
    return abs(created - float(started_at)) <= PID_IDENTITY_WINDOW_S


def kill_tree(pid: int) -> None:
    """프로세스와 그 자식(DDP rank · 데이터로더 워커)까지 정리한다.

    호출 전에 is_our_worker 로 신원을 확인할 것. 여기서는 확인하지 않는다.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False
        )
        return
    try:
        os.kill(int(pid), 9)
    except OSError:
        pass
