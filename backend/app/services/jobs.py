"""학습이 아닌 백그라운드 작업의 공통 관리 — 기동·상태·정지·재시작 복구.

내보내기 하나뿐일 때는 run_manager 안에 전용 코드로 두어도 됐다. 그런데 학습 후 진단,
데이터셋 임베딩처럼 같은 성질의 작업이 이어서 붙는다. 전부 독립 프로세스로 돌고, 오래 걸리고,
진행 상황을 jsonl 로 흘리고, 백엔드가 재시작되면 PID 로 다시 붙잡아야 하고, GPU 를 쓰면
학습과 같은 슬롯을 두고 경쟁한다.

그 공통부를 여기 한 곳에 둔다. 잡을 새로 추가하는 쪽은 SPECS 에 한 줄과 워커 스크립트만
쓰면 되고, 이 파일은 더 건드리지 않는다.

**GPU 중재는 여기서 하지 않는다.** "지금 GPU 가 비었나" 는 학습 큐를 아는 run_manager 만
답할 수 있고, run_manager 는 이미 이 모듈을 쓴다. 양쪽이 서로를 import 하면 순환이 된다.
그래서 여기는 프로세스 레지스트리로만 두고, 슬롯 배정은 run_manager.start_job 이 맡는다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from app.core import db, procs
from app.core.config import BACKEND_DIR, DATASETS_DIR, RUNS_DIR, WEIGHTS_DIR, offline_env

OWNER_ROOTS: dict[str, Path] = {"run": RUNS_DIR, "dataset": DATASETS_DIR}


class JobError(Exception):
    """사용자에게 그대로 보여줄 수 있는 실패."""


@dataclass(frozen=True)
class JobSpec:
    kind: str
    owner_type: str
    label: str  # 한국어 표시명 ("내보내기")
    script: str  # backend/<script>
    validate: Callable[[dict[str, Any]], dict[str, Any]]
    needs_gpu: Callable[[dict[str, Any]], bool]
    build_argv: Callable[[Path, Path, dict[str, Any], list[int]], list[str]]
    # GPU 를 못 잡았을 때 CPU 로 내려서라도 돌릴 것인가.
    # TensorRT 변환처럼 GPU 가 없으면 아예 불가능한 작업은 False 로 두고 거절한다.
    gpu_optional: bool = False


SPECS: dict[str, JobSpec] = {}


def register(spec: JobSpec) -> None:
    SPECS[spec.kind] = spec


# ------------------------------------------------------------------ 내보내기

EXPORT_FORMATS = {"onnx", "torchscript", "engine"}
# GPU 가 필요한 포맷. 나머지는 CPU 로 변환하므로 학습과 경합하지 않는다.
GPU_FORMATS = {"engine"}


def _is_contained(relative: str) -> bool:
    """소유자 폴더 안을 가리키는 상대 경로인가.

    Path.is_absolute() 만으로는 부족하다. Windows 에서 "/etc/passwd" 는 드라이브 문자가
    없어 is_absolute() 가 False 인데, run_dir / "/etc/passwd" 는 C:/etc/passwd 가 되어
    폴더를 벗어난다. 드라이브와 루트를 둘 다 본다.
    """
    if not relative:
        return False
    for flavour in (PureWindowsPath, PurePosixPath):
        candidate = flavour(relative)
        if candidate.drive or candidate.root or ".." in candidate.parts:
            return False
    return True


def _validate_export(args: dict[str, Any]) -> dict[str, Any]:
    fmt = str(args.get("format", "onnx"))
    if fmt not in EXPORT_FORMATS:
        raise JobError(f"지원하지 않는 포맷입니다: {fmt}")
    weights = str(args.get("weights", "train/weights/best.pt"))
    if not _is_contained(weights):
        # 소유자 폴더 밖의 파일을 내보내지 못하게 한다. 경로는 상대로만 저장한다 —
        # 절대 경로를 넣으면 폴더를 옮겼을 때 깨진다(scripts/relocate.py 가 손대지 않는다).
        raise JobError("실행 폴더 안의 상대 경로만 지정할 수 있습니다.")
    return {
        "format": fmt,
        "weights": weights,
        "imgsz": int(args.get("imgsz", 640)),
        "half": bool(args.get("half", False)),
        "dynamic": bool(args.get("dynamic", False)),
    }


def _argv_export(
    owner: Path, directory: Path, args: dict[str, Any], devices: list[int]
) -> list[str]:
    argv = [
        "--run-dir", str(owner),
        "--events", str(directory / "events.jsonl"),
        "--format", args["format"],
        "--weights", args["weights"],
        "--imgsz", str(args["imgsz"]),
        "--device", ",".join(str(d) for d in devices) if devices else "cpu",
    ]
    if args["half"]:
        argv.append("--half")
    if args["dynamic"]:
        argv.append("--dynamic")
    return argv


def _validate_analyze(args: dict[str, Any]) -> dict[str, Any]:
    weights = str(args.get("weights", "train/weights/best.pt"))
    if not _is_contained(weights):
        raise JobError("실행 폴더 안의 상대 경로만 지정할 수 있습니다.")
    imgsz = int(args.get("imgsz", 640))
    if not 32 <= imgsz <= 4096:
        raise JobError("이미지 크기는 32~4096 이어야 합니다.")
    return {
        "weights": weights,
        "imgsz": imgsz - imgsz % 32,
        "batch": max(1, min(int(args.get("batch", 8)), 64)),
        # 요청이 GPU 를 직접 고르지 못하게 한다. 학습이 쓰는 GPU 에 얹으면 학습이 죽는다 —
        # 비어 있을 때만 서버가 배정한다.
        "use_gpu": bool(args.get("use_gpu", False)),
    }


def _argv_analyze(
    owner: Path, directory: Path, args: dict[str, Any], devices: list[int]
) -> list[str]:
    return [
        "--run-dir", str(owner),
        "--out-dir", str(directory),
        "--events", str(directory / "events.jsonl"),
        "--weights", args["weights"],
        "--imgsz", str(args["imgsz"]),
        "--batch", str(args["batch"]),
        "--device", ",".join(str(d) for d in devices) if devices else "cpu",
    ]


register(
    JobSpec(
        kind="analyze",
        owner_type="run",
        label="오류 분석",
        script="analysis_worker.py",
        validate=_validate_analyze,
        # GPU 는 사용자가 켜고 비어 있을 때만 쓴다. 못 잡으면 CPU 로 도는 게 맞다 —
        # 분석은 배치 작업이라 조금 느려도 되지만, 학습을 죽이면 안 된다.
        needs_gpu=lambda args: args["use_gpu"],
        build_argv=_argv_analyze,
        gpu_optional=True,
    )
)


def _validate_quality(args: dict[str, Any]) -> dict[str, Any]:
    # int(...) 를 그냥 부르면 "bad" 가 ValueError 로 새어 나가 422 가 아니라 500 이 된다.
    try:
        imgsz = int(args.get("imgsz", 224))
    except (TypeError, ValueError) as exc:
        raise JobError("이미지 크기는 숫자여야 합니다.") from exc
    if not 64 <= imgsz <= 640:
        raise JobError("이미지 크기는 64~640 이어야 합니다.")
    return {
        "imgsz": imgsz - imgsz % 32,
        # analyze 와 같은 이유로 GPU 번호를 요청이 직접 고르지 못하게 한다.
        "use_gpu": bool(args.get("use_gpu", False)),
    }


def _argv_quality(
    owner: Path, directory: Path, args: dict[str, Any], devices: list[int]
) -> list[str]:
    # 원본 폴더(root) 는 넘기지 않는다 — 이미지 목록의 진실은 owner 폴더 안의
    # train.txt / val.txt 다. quality_worker.py 첫머리 주석 참고.
    return [
        "--dataset-dir", str(owner),
        "--out-dir", str(directory),
        "--events", str(directory / "events.jsonl"),
        "--imgsz", str(args["imgsz"]),
        "--device", ",".join(str(d) for d in devices) if devices else "cpu",
    ]


register(
    JobSpec(
        kind="quality",
        owner_type="dataset",
        label="데이터 품질 검사",
        script="quality_worker.py",
        validate=_validate_quality,
        # 분석 잡과 같은 정책: 사용자가 켜고 GPU 가 비었을 때만 쓴다. 못 잡으면 CPU.
        needs_gpu=lambda args: args["use_gpu"],
        build_argv=_argv_quality,
        gpu_optional=True,
    )
)


register(
    JobSpec(
        kind="export",
        owner_type="run",
        label="내보내기",
        script="export_worker.py",
        validate=_validate_export,
        needs_gpu=lambda args: args["format"] in GPU_FORMATS,
        build_argv=_argv_export,
    )
)


def spec_for(kind: str) -> JobSpec:
    spec = SPECS.get(kind)
    if spec is None:
        raise JobError(f"알 수 없는 작업 종류입니다: {kind}")
    return spec


# ------------------------------------------------------------------ 경로


def owner_dir(owner_type: str, owner_id: str) -> Path:
    root = OWNER_ROOTS.get(owner_type)
    if root is None:
        raise JobError(f"알 수 없는 소유자 종류입니다: {owner_type}")
    # 경로 조각을 그대로 붙이면 ../ 로 루트 밖을 가리킬 수 있다.
    resolved = (root / owner_id).resolve()
    if root.resolve() not in resolved.parents:
        raise JobError("잘못된 소유자 ID 입니다.")
    return resolved


def job_dir(kind: str, owner_type: str, owner_id: str) -> Path:
    """산출물은 소유자 폴더 아래에 둔다.

    run/dataset 을 지울 때 fsops.remove_tree 가 이미 하위 전체를 지우므로
    잡 산출물을 따로 정리하는 코드가 필요 없다.
    """
    return owner_dir(owner_type, owner_id) / "jobs" / kind


def events_path(kind: str, owner_type: str, owner_id: str) -> Path:
    return job_dir(kind, owner_type, owner_id) / "events.jsonl"


# ------------------------------------------------------------------ 상태


def _row(kind: str, owner_type: str, owner_id: str) -> dict[str, Any] | None:
    row = db.query_one(
        "SELECT * FROM jobs WHERE kind = ? AND owner_type = ? AND owner_id = ?"
        " ORDER BY created_at DESC LIMIT 1",
        (kind, owner_type, owner_id),
    )
    return db.row_to_job(row) if row else None


def alive(job: dict[str, Any]) -> bool:
    return job["status"] == "running" and procs.is_our_worker(job["pid"], job["started_at"])


def reserved_devices() -> set[int]:
    """지금 잡이 붙잡고 있는 GPU.

    프로세스를 분리해도 VRAM 은 분리되지 않는다. 여기서 빠뜨리면 스케줄러가 그 GPU 위에
    학습을 띄워 둘 다 OOM 난다.
    """
    used: set[int] = set()
    for row in db.query("SELECT * FROM jobs WHERE status = 'running'"):
        job = db.row_to_job(row)
        if alive(job):
            used.update(job["devices"])
    return used


def live_for(owner_type: str, owner_id: str) -> list[dict[str, Any]]:
    """이 소유자에게 붙어 실제로 돌고 있는 잡. 삭제 가드가 쓴다."""
    rows = db.query(
        "SELECT * FROM jobs WHERE owner_type = ? AND owner_id = ? AND status = 'running'",
        (owner_type, owner_id),
    )
    return [job for job in (db.row_to_job(r) for r in rows) if alive(job)]


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return events


def status(kind: str, owner_type: str, owner_id: str) -> dict[str, Any]:
    job = _row(kind, owner_type, owner_id)
    if job is None:
        return {
            "status": "idle",
            "kind": kind,
            "owner_type": owner_type,
            "owner_id": owner_id,
            "args": {},
            "devices": [],
            "events": [],
            "result": None,
            "error": None,
        }

    events = _read_events(events_path(kind, owner_type, owner_id))
    end = next((e for e in reversed(events) if e.get("t") == "end"), None)
    state = job["status"]
    if state == "running" and not alive(job):
        # 프로세스가 사라졌다. end 이벤트가 있으면 그걸 따르고, 없으면 죽은 것으로 확정한다.
        state = end.get("status", "completed") if end else "failed"
        error = None if end else "작업 프로세스가 결과를 남기지 못하고 종료되었습니다."
        _finish(job["id"], state, error)
        job["error"] = job["error"] or error
    return {**job, "status": state, "events": events, "result": end}


# ------------------------------------------------------------------ 수명주기


def _finish(job_id: str, state: str, error: str | None = None) -> None:
    db.execute(
        "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
        (state, error, time.time(), job_id),
    )


def spawn(
    kind: str,
    owner_type: str,
    owner_id: str,
    args: dict[str, Any],
    devices: list[int],
) -> dict[str, Any]:
    """워커를 띄운다. GPU 예약 판단은 호출자(run_manager)가 이미 끝냈다고 본다."""
    spec = spec_for(kind)
    if spec.owner_type != owner_type:
        raise JobError(f"{spec.label} 는 {spec.owner_type} 에만 붙일 수 있습니다.")

    directory = job_dir(kind, owner_type, owner_id)
    directory.mkdir(parents=True, exist_ok=True)
    # 이전 실행의 이벤트가 남아 있으면 새 실행의 진행 상황과 섞인다.
    events_path(kind, owner_type, owner_id).unlink(missing_ok=True)

    job_id = uuid.uuid4().hex[:12]
    started_at = time.time()
    db.execute(
        "INSERT INTO jobs (id,kind,owner_type,owner_id,status,args,devices,created_at,started_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (job_id, kind, owner_type, owner_id, "running", json.dumps(args, default=str),
         json.dumps(devices), started_at, started_at),
    )

    env = os.environ.copy()
    env.update(offline_env())
    env["PYTHONUNBUFFERED"] = "1"
    argv = [sys.executable, str(BACKEND_DIR / spec.script)]
    argv += spec.build_argv(owner_dir(owner_type, owner_id), directory, args, devices)

    try:
        log_file = open(directory / "job.log", "ab", buffering=0)
        process = subprocess.Popen(
            argv,
            cwd=str(WEIGHTS_DIR if WEIGHTS_DIR.is_dir() else BACKEND_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except Exception as exc:  # noqa: BLE001 - 못 띄웠으면 행을 남겨 두지 않는다
        _finish(job_id, "failed", f"작업을 시작하지 못했습니다: {exc}")
        raise JobError(f"작업을 시작하지 못했습니다: {exc}") from exc

    db.execute("UPDATE jobs SET pid = ? WHERE id = ?", (process.pid, job_id))
    return status(kind, owner_type, owner_id)


def stop(kind: str, owner_type: str, owner_id: str) -> None:
    job = _row(kind, owner_type, owner_id)
    if job is None or job["status"] != "running":
        return
    if procs.is_our_worker(job["pid"], job["started_at"]):
        procs.kill_tree(int(job["pid"]))
    _finish(job["id"], "stopped")


def sweep() -> None:
    """죽은 잡의 상태를 확정한다.

    아무도 상태를 조회하지 않아도 GPU 예약이 풀려야 하므로 스케줄러가 주기적으로 부른다.
    """
    for row in db.query("SELECT * FROM jobs WHERE status = 'running'"):
        job = db.row_to_job(row)
        if not alive(job):
            status(job["kind"], job["owner_type"], job["owner_id"])


def recover() -> None:
    """백엔드 재시작 시 running 으로 남아 있던 잡을 정리한다.

    프로세스가 아직 살아 있으면 그대로 두고(PID 로 계속 추적된다), 아니면 확정한다.
    """
    _adopt_legacy_exports()
    sweep()


def _adopt_legacy_exports() -> None:
    """jobs 테이블 이전에 쓰던 export.job.json 을 흡수한다.

    이관 시점에 내보내기가 돌고 있으면 새 코드가 그 GPU 예약을 못 본다 → 스케줄러가 같은
    GPU 에 학습을 띄워 둘 다 OOM 난다. 확률은 낮지만 결과가 크다.

    다음 릴리스에서 제거할 것. 그때는 이 파일 형식이 더 이상 만들어지지 않는다.
    """
    if not RUNS_DIR.is_dir():
        return
    for path in RUNS_DIR.glob("*/export.job.json"):
        run_id = path.parent.name
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            continue

        pid, started_at = record.get("pid"), record.get("started_at")
        if procs.is_our_worker(pid, started_at):
            db.execute(
                "INSERT INTO jobs"
                " (id,kind,owner_type,owner_id,status,args,devices,pid,created_at,started_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex[:12], "export", "run", run_id, "running",
                 json.dumps({"format": record.get("format", "")}),
                 json.dumps(record.get("devices", [])), pid, started_at, started_at),
            )
        path.unlink(missing_ok=True)
