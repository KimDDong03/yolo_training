"""실패한 학습을 원인·처방으로 번역하고, 고친 파라미터로 다시 돌릴 준비를 한다.

지금까지 실패는 runs.error 에 원문 한 줄로만 남았다. "CUDA out of memory. Tried to
allocate 2.00 GiB" 를 받아 든 비전문가는 다음에 무엇을 바꿔야 하는지 알 수 없다.

진단은 **요청받을 때마다 다시 계산한다.** 실패 시점에 DB 로 굳히지 않는 이유:
근거 파일(events.jsonl · train.log)이 run 폴더에 영구히 남아 언제든 재계산할 수 있고,
규칙을 개선하면 과거 실패에도 즉시 적용되기 때문이다. 굳혀 두면 옛 진단이 영원히 남는다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.core import db
from app.services import run_manager

LOG_TAIL_LINES = 200

# params 와 devices 를 받아 고친 사본을 돌려준다. 무엇이 바뀌었는지는 호출부가 diff 로 뽑는다.
Patch = Callable[[dict[str, Any], list[int]], tuple[dict[str, Any], list[int]]]


@dataclass(frozen=True)
class Rule:
    code: str
    pattern: re.Pattern[str]
    title: str
    cause: str
    fix: str
    patch: Patch | None = None
    patch_label: str = ""
    # 같은 설정으로 다시 돌려서 결과가 달라질 여지가 있는가.
    # 라벨이 없거나 데이터가 사라진 경우는 사용자가 밖에서 고치기 전에는 똑같이 실패한다 —
    # 그때 재시도 버튼을 주면 "눌러도 또 실패"를 반복시키는 것뿐이다.
    retryable: bool = True


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _oom(params: dict[str, Any], devices: list[int]) -> tuple[dict[str, Any], list[int]]:
    out = dict(params)
    batch = out.get("batch", -1)
    # -1(자동)로 이미 터졌다면 자동 산정이 과했다는 뜻이라 고정값으로 내려 잡는다.
    out["batch"] = 8 if batch in (-1, None) else int(_clamp(int(batch) // 2, 1, 1024))
    if not out.get("amp", True):
        out["amp"] = True
    return out, devices


def _nan(params: dict[str, Any], devices: list[int]) -> tuple[dict[str, Any], list[int]]:
    out = dict(params)
    # 혼합정밀에서 fp16 언더/오버플로로 손실이 발산하는 경우가 가장 흔하다.
    out["amp"] = False
    out["lr0"] = _clamp(float(out.get("lr0", 0.01)) / 2, 1e-6, 1.0)
    return out, devices


def _single_gpu(params: dict[str, Any], devices: list[int]) -> tuple[dict[str, Any], list[int]]:
    return dict(params), devices[:1]


def _fewer_workers(params: dict[str, Any], devices: list[int]) -> tuple[dict[str, Any], list[int]]:
    out = dict(params)
    out["workers"] = int(_clamp(min(int(out.get("workers", 8) or 0), 2), 0, 32))
    return out, devices


# 위에서부터 첫 매치를 채택한다 — 구체적인 것을 앞에 둔다.
#
# 한글 Windows 주의: OSError/WinError 계열은 설명 문자열이 한국어로 나온다.
# 그래서 패턴은 영문 예외 클래스명과 WinError 번호를 기준으로 짠다. 한국어 문구에 의존하면
# 로케일이 다른 PC 에서 조용히 안 맞는다.
RULES: list[Rule] = [
    Rule(
        "cuda_oom",
        re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED", re.I),
        "GPU 메모리가 부족했습니다",
        "지금 설정으로는 이 GPU 에 모델과 배치가 함께 올라가지 않습니다.",
        "배치 크기를 줄이는 것이 가장 효과가 큽니다. 이미지 크기를 낮추거나 AMP(혼합정밀)를 "
        "켜도 VRAM 사용이 줄어듭니다.",
        _oom,
        "배치를 줄이고 AMP 를 켜서 재시작",
    ),
    Rule(
        "nan_loss",
        re.compile(r"\bnan\b[^\n]{0,40}loss|loss[^\n]{0,40}\bnan\b|NaN or Inf found", re.I),
        "손실이 NaN 으로 발산했습니다",
        "학습률이 너무 높거나 혼합정밀(AMP)에서 값이 넘쳐 손실이 계산 불가능해졌습니다.",
        "학습률을 낮추고 AMP 를 꺼서 다시 시도하세요. 그래도 같으면 데이터의 라벨 좌표를 "
        "검수 화면에서 확인하세요.",
        _nan,
        "학습률을 절반으로 낮추고 AMP 를 꺼서 재시작",
    ),
    Rule(
        "no_labels",
        re.compile(r"No labels found|No images found|labels_missing|train:.*0 images", re.I),
        "학습할 라벨을 찾지 못했습니다",
        "데이터셋에서 이미지와 짝이 맞는 라벨 파일을 읽지 못했습니다.",
        "데이터셋 검수 화면에서 '라벨 없는 이미지' 를 확인하세요. images/ 와 labels/ 의 "
        "폴더 구조와 파일 이름이 서로 맞아야 합니다.",
        retryable=False,
    ),
    Rule(
        "data_missing",
        re.compile(r"데이터셋 정의를 찾을 수 없습니다|Dataset .{0,80} not found|"
                   r"can'?t open file .{0,40}\.yaml", re.I),
        "데이터셋 파일이 사라졌습니다",
        "학습을 시작할 때 있던 data.yaml 또는 이미지 폴더가 지금은 없습니다.",
        "데이터셋을 옮겼거나 지웠는지 확인하세요. 폴더를 통째로 옮겼다면 "
        "scripts/relocate.py 를 실행해야 합니다.",
        retryable=False,
    ),
    Rule(
        "model_missing",
        re.compile(r"모델 이름입니다|파일을 찾을 수 없습니다|"
                   r"FileNotFoundError[^\n]{0,80}\.(pt|yaml)|No such file[^\n]{0,80}\.pt", re.I),
        "모델 가중치를 찾지 못했습니다",
        "지정한 .pt / .yaml 경로가 지금은 존재하지 않습니다.",
        "원본 가중치 경로로 되돌려 다시 시도합니다. 이전 학습의 best.pt 를 썼다면 그 학습을 "
        "지우지 않았는지 확인하세요.",
    ),
    Rule(
        "download_blocked",
        re.compile(r"Downloading https|Max retries exceeded|urlopen error|"
                   r"getaddrinfo failed|ConnectionError", re.I),
        "외부 다운로드를 시도하다 막혔습니다",
        "단독망이라 인터넷에 나갈 수 없는데 ultralytics 가 가중치나 폰트를 받으려 했습니다. "
        "보통 번들에 없는 모델 이름을 지정했을 때 생깁니다.",
        "bundle/weights/ 에 실제로 있는 .pt 파일의 경로를 직접 지정하세요.",
        retryable=False,
    ),
    Rule(
        "ddp_failed",
        re.compile(r"torch\.distributed|NCCL|ProcessGroup|DDP|gloo", re.I),
        "다중 GPU(DDP) 학습이 실패했습니다",
        "여러 GPU 를 함께 쓰는 분산 학습이 초기화되지 못했습니다.",
        "GPU 한 장으로 먼저 학습이 되는지 확인하세요.",
        _single_gpu,
        "GPU 한 장으로 재시작",
    ),
    Rule(
        "worker_crash",
        re.compile(r"DataLoader worker|PermissionError|WinError 5\b|"
                   r"page file is too small|WinError 1455", re.I),
        "데이터 로더 프로세스가 죽었습니다",
        "Windows 에서 데이터 로딩 프로세스를 너무 많이 띄우면 가상 메모리나 권한 문제로 "
        "죽는 경우가 있습니다.",
        "데이터 로더 워커 수를 줄여 다시 시도하세요.",
        _fewer_workers,
        "데이터 로더 워커를 2개로 줄여 재시작",
    ),
    Rule(
        "disk_full",
        re.compile(r"No space left|WinError 112|disk full|OSError 28", re.I),
        "디스크 공간이 부족합니다",
        "학습 산출물을 저장할 공간이 남아 있지 않습니다.",
        "storage/runs 에서 오래된 학습 기록을 지우거나 다른 드라이브를 확보하세요.",
    ),
]


def _read_tail(path: Path, lines: int) -> list[str]:
    """텍스트 파일의 마지막 N 줄. 인코딩이 섞여 있어도 죽지 않아야 한다.

    워커 stdout 은 한국어 Windows 에서 cp949 와 utf-8 이 섞인다 (기존 관례대로 replace).
    """
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def _last_end_event(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return {}
    last: dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("t") == "end":
                    last = obj
    except OSError:
        return {}
    return last


def _source_model(run_dir: Path) -> str | None:
    """사용자가 원래 고른 가중치 경로.

    params.model 은 run 폴더 안의 사본(inputs/*.pt)을 가리킨다(run_manager.create_run).
    그걸 그대로 재시도에 쓰면, 이 실패 run 을 지운 순간 재시도로 만든 run 도 깨진다.
    """
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return config.get("source_model") or None


def _diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"from": before.get(key), "to": after[key]}
        for key in after
        if before.get(key) != after[key]
    }


def diagnose(run_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        raise run_manager.RunError("실행을 찾을 수 없습니다.")
    run = db.row_to_run(row)
    run_dir = run_manager.run_dir_for(run_id)

    log_tail = _read_tail(run_dir / "train.log", LOG_TAIL_LINES)
    result: dict[str, Any] = {
        "run_id": run_id,
        "status": run["status"],
        "matched": False,
        "code": None,
        "title": None,
        "cause": None,
        "fix": None,
        "evidence": [],
        "log_tail": log_tail,
        "retry": None,
    }
    if run["status"] != "failed":
        return result

    end = _last_end_event(run_dir)
    # 순서가 중요하다. 워커가 end 이벤트를 남기지 못하고 죽으면 진짜 메시지가 train.log 에만 있다
    # (run_manager._settle 이 그 경우를 대신 메운다).
    haystack = "\n".join(
        part
        for part in (run["error"], end.get("error"), end.get("traceback"), "\n".join(log_tail))
        if part
    )

    rule = next((r for r in RULES if r.pattern.search(haystack)), None)
    if rule is None:
        # runs.error 가 비어 있어도 end 이벤트에는 남아 있을 수 있다 — 워커가 죽어
        # 백엔드가 대신 종료를 기록한 경우가 그렇다. 손에 든 메시지를 버리지 않는다.
        detail = next((t for t in (run["error"], end.get("error")) if t), None)
        result["title"] = "학습이 예기치 않게 종료되었습니다"
        result["cause"] = detail or "종료 원인을 특정하지 못했습니다."
        result["fix"] = "아래 로그 원문에서 마지막 오류 줄을 확인하세요."
        return result

    result.update(
        matched=True,
        code=rule.code,
        title=rule.title,
        cause=rule.cause,
        fix=rule.fix,
        evidence=[line for line in haystack.splitlines() if rule.pattern.search(line)][-5:],
    )

    if not rule.retryable:
        return result

    params, devices = dict(run["params"]), list(run["devices"])
    source_model = _source_model(run_dir)
    if source_model:
        params["model"] = source_model
    if rule.patch is not None:
        params, devices = rule.patch(params, devices)

    changed = _diff(run["params"], params)
    # model 복원은 사용자가 바꾼 설정이 아니라 내부 경로 정리라 변경 목록에서 감춘다.
    changed.pop("model", None)
    # devices 는 params 밖에 있다. 여기서 안 넣으면 "GPU 한 장으로 재시작" 이 변경 목록에
    # 아무것도 없는 채로 보인다 — 실제로는 바뀌는데.
    if devices != run["devices"]:
        changed["devices"] = {"from": run["devices"], "to": devices}
    result["retry"] = {
        "label": rule.patch_label or "같은 설정으로 재시작",
        "changed": changed,
        "params": params,
        "options": run["options"],
        "devices": devices,
    }
    return result
