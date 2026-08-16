"""하이퍼파라미터 자동 탐색 워커 — ultralytics `model.tune()`.

다른 워커와 같은 규약을 따른다: 독립 프로세스, argparse, events.jsonl 에 append.
다만 이 잡만의 사정이 셋 있고, 셋 다 ultralytics 원본을 읽고 맞춘 것이다.

1. **시도는 별도 subprocess 로 돈다.** `Tuner.__call__` 은 매 반복
   `python -m ultralytics.cfg.__init__ train k=v ...` 를 띄운다. 부모에 붙인 콜백은 전달되지
   않지만 hooks/sitecustomize.py 는 PYTHONPATH 를 타고 붙으므로(DDP 와 같은 원리),
   YOLOWEB_RUN_DIR 만 넘기면 시도 안의 epoch 진행이 그대로 흐른다.

2. **시도 이벤트는 잡 이벤트와 다른 파일에 쓴다**(`<out_dir>/trials/events.jsonl`).
   훅은 학습이 끝날 때마다 `{"t":"end"}` 를 쓰는데, jobs.status() 는 마지막 end 이벤트를
   잡의 결과로 읽는다(jobs.py:424). 한 파일에 섞으면 **워커가 도중에 죽어도 마지막 시도의
   end 때문에 잡이 "완료" 로 보인다.** 파일을 나누면 그 혼선이 원천적으로 없고, 잡
   events.jsonl 을 쓰는 것은 이 워커 하나뿐이라 동시 append 도 생기지 않는다.

3. **`resume=True` 로 고정한다.** Tuner 는 `exist_ok = resume` 로 두고 get_save_dir 는
   exist_ok 가 거짓이면 경로를 증가시킨다(tune → tune2). resume 을 안 주면 두 번째 실행이
   tune2 로 밀려 이어하기가 깨지고 폴러가 볼 경로도 어긋난다. "처음부터" 는 resume 인자가
   아니라 **디렉터리를 지우는 것**으로 표현한다(--restart).

진행 상황은 tune.json 하나로 모은다. 실행 중에도 화면이 이 파일을 읽으므로 임시 파일에 쓰고
os.replace 로 바꿔 끼운다 — 반쯤 쓰인 JSON 을 읽는 일이 없어야 한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import BACKEND_DIR, HOOKS_DIR, apply_offline_env  # noqa: E402

apply_offline_env()

from app.services import tune  # noqa: E402  (offline 환경을 잡은 뒤에 가져온다)

POLL_INTERVAL = 2.0
# 시도 진행을 읽을 때 훑는 꼬리 크기. 배치 이벤트가 0.5초마다 쌓여 파일이 계속 자라므로
# 전체를 읽지 않는다. 진행 표시용이라 조금 놓쳐도 된다.
TAIL_BYTES = 256 * 1024


def write(path: Path, payload: dict) -> None:
    payload.setdefault("ts", time.time())
    # 한 줄을 한 번의 write 로 내보내야 리더가 잘린 JSON 을 보지 않는다.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def write_json(path: Path, payload: dict) -> None:
    """원자적으로 바꿔 끼운다. 화면이 실행 중에 이 파일을 읽는다.

    `allow_nan=False` 는 마지막 방어선이다. 파이썬 json 은 기본으로 표준이 아닌 `NaN` 리터럴을
    쓰고 브라우저의 `JSON.parse` 가 거기서 죽는다. 값은 tune._finite 가 이미 걸렀으므로
    여기서 걸릴 일이 없어야 하고, 걸린다면 조용히 넘어가는 대신 드러나는 편이 맞다.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def tail_events(path: Path) -> list[dict]:
    """시도 이벤트 파일의 꼬리만 읽는다."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    try:
        with open(path, "rb") as fh:
            if size > TAIL_BYTES:
                fh.seek(size - TAIL_BYTES)
                fh.readline()  # 잘린 첫 줄은 버린다
            chunk = fh.read()
    except OSError:
        return []
    events: list[dict] = []
    for line in chunk.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


SIGNATURE_NAME = "yoloweb_signature.json"
NOISE_NAME = "yoloweb_noise.json"


def _train_once(save_dir: Path, train_args: dict) -> float | None:
    """Tuner 가 시도를 돌리는 방식 그대로 한 번 학습하고 fitness 를 돌려준다.

    같은 방식(subprocess + 체크포인트의 train_metrics)이라야 시도와 같은 조건이 된다
    (tuner.py:492-497).
    """
    shutil.rmtree(save_dir, ignore_errors=True)
    cmd = [sys.executable, "-m", "ultralytics.cfg.__init__", "train"]
    cmd += [f"{k}={v}" for k, v in {**train_args, "save_dir": str(save_dir)}.items()]
    proc = subprocess.run(cmd, check=False)
    weights = save_dir / "weights"
    ckpt = weights / ("best.pt" if (weights / "best.pt").exists() else "last.pt")
    if proc.returncode != 0 or not ckpt.is_file():
        return None
    try:
        from ultralytics.utils.patches import torch_load

        fitness = float(torch_load(ckpt)["train_metrics"]["fitness"])
    except Exception:  # noqa: BLE001 - 못 재면 바닥선으로 돌아간다. 탐색을 죽이지는 않는다.
        return None
    return fitness if math.isfinite(fitness) else None


def measure_noise(
    tune_dir: Path,
    baseline: dict,
    base_args: dict,
    data: str,
    device: str,
    announce=None,
) -> dict | None:
    """기준 하이퍼파라미터를 **시드만 바꿔** 여러 번 돌려 흔들림의 표준편차를 잰다.

    왜 필요한가: 시도는 전부 seed 0 으로 돌아 같은 값이면 결과가 완전히 재현된다. 그래서 탐색은
    자기가 낸 상승폭이 하이퍼파라미터 덕인지 시드 운인지 스스로 구분할 수 없다.
    실측(.codex/phase-4.md)에서 같은 값도 시드만 바꾸면 에폭 3·데이터 15% 에서 표준편차
    0.026, 에폭 10·데이터 100% 에서 0.007 로 **3.5배** 달랐다. 상수 하나로는 어느 한쪽에서
    반드시 틀린다. 그래서 사용자의 설정 그대로 다시 돌려 그 자리에서 잰다.

    **한 번으로는 부족하다.** 처음에는 시드 하나만 다시 돌려 그 차이를 문턱으로 썼는데,
    실측에서 그 한 번이 우연히 가깝게 나와(0.0082, 진짜 표준편차는 0.0258) 정확히 걸러야 할
    상승(+0.0189)을 통과시켰다. 기준까지 합쳐 네 표본이라야 표준편차를 추정할 수 있다.

    캐시한다 — 기준값과 설정이 그대로면(서명이 지킨다) 다시 잴 이유가 없다.
    """
    cached = tune_dir / NOISE_NAME
    if cached.is_file():
        try:
            return json.loads(cached.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    hyp = {key: value for key, value in baseline["hyp"].items() if key in tune.SPACE}
    measured: list[float] = []
    for index, seed in enumerate(tune.PROBE_SEEDS):
        if announce:
            announce(index + 1, len(tune.PROBE_SEEDS))
        fitness = _train_once(
            tune_dir / f"noise{seed}",
            {**base_args, **hyp, "data": data, "device": device, "seed": seed,
             "plots": False, "save_period": -1, "exist_ok": True},
        )
        if fitness is not None:
            measured.append(round(fitness, 5))

    # 기준(시드 0)까지 합쳐야 표본이다. 확인 시도가 하나도 못 돌았으면 잴 것이 없다.
    samples = [baseline["fitness"], *measured]
    if len(samples) < 3:
        return None

    result = {
        "seeds": list(tune.PROBE_SEEDS)[: len(measured)],
        "fitness": measured,
        "baseline_fitness": baseline["fitness"],
        "stdev": round(statistics.stdev(samples), 5),
        "range": round(max(samples) - min(samples), 5),
    }
    try:
        cached.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return result


def signature_of(tune_dir: Path) -> dict | None:
    """이 기록이 어떤 설정으로 만들어졌는지. 없으면 None."""
    path = tune_dir / SIGNATURE_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def current_trial(trials_dir: Path) -> dict[str, Any] | None:
    """지금 도는 시도의 에폭 진행. 없으면 None.

    에폭 번호는 시도마다 1부터 다시 시작하므로 **마지막 start 이벤트 뒤쪽만** 본다.
    """
    events = tail_events(trials_dir / "events.jsonl")
    if not events:
        return None
    begin = 0
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("t") == "start":
            begin = index
            break
    window = events[begin:]
    if any(e.get("t") == "end" for e in window):
        # 이 시도는 끝났고 다음 시도는 아직 시작하지 않았다. 마지막 에폭을 그대로 두면
        # 준비 중인 시간 내내 "에폭 5/5" 가 진행 중인 것처럼 보인다.
        return None
    start = window[0] if window and window[0].get("t") == "start" else {}
    epoch = next((e for e in reversed(window) if e.get("t") == "epoch"), None)
    batch = next((e for e in reversed(window) if e.get("t") == "batch"), None)
    total = start.get("total_epochs") or (epoch or {}).get("total_epochs")
    if not total:
        return None
    return {
        "epoch": (epoch or {}).get("epoch") or 0,
        "total_epochs": total,
        "batch": (batch or {}).get("i"),
        "batch_total": (batch or {}).get("n"),
    }


class Poller(threading.Thread):
    """tune_results.ndjson 을 지켜보다 tune.json 을 갱신한다.

    Tuner 는 반복 단위 콜백을 부르지 않아(원본 확인) 이것 말고는 진행을 알 방법이 없다.
    """

    def __init__(
        self, out_dir: Path, tune_dir: Path, events: Path, job_args: dict, resumed: int
    ) -> None:
        super().__init__(daemon=True)
        self.out_dir = out_dir
        self.tune_dir = tune_dir
        self.events = events
        self.job_args = job_args
        self.resumed = resumed
        self.started = time.time()
        self.stopping = threading.Event()
        self.done = 0
        # 확인 시도가 끝나면 채워진다. 그전까지는 바닥선만으로 판정한다.
        self.noise: dict | None = None

    def snapshot(self) -> dict:
        # 남은 시간은 시작 전 추정이 아니라 이 잡이 실제로 쓴 시간으로 낸다.
        report = tune.build_report(
            self.tune_dir,
            self.job_args,
            elapsed_s=time.time() - self.started,
            resumed=self.resumed,
            noise=self.noise,
        )
        report["current"] = current_trial(self.out_dir / "trials")
        write_json(self.out_dir / "tune.json", report)
        return report

    def run(self) -> None:
        while not self.stopping.wait(POLL_INTERVAL):
            try:
                report = self.snapshot()
            except Exception:  # noqa: BLE001 - 진행 표시가 탐색을 죽이면 안 된다
                continue
            done = int(report.get("iterations_done") or 0)
            if done > self.done:
                self.done = done
                best = report.get("best") or {}
                # 이 파일을 쓰는 프로세스는 이 워커 하나뿐이다(시도는 trials/ 로 나갔다).
                write(
                    self.events,
                    {
                        "t": "progress",
                        "stage": "trial",
                        "message": f"시도 {done}/{report.get('iterations_target')} 완료"
                        f" · 최고 fitness {float(best.get('fitness') or 0.0):.4f}",
                    },
                )

    def stop(self) -> None:
        self.stopping.set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events = Path(args.events).resolve()
    tune_dir = out_dir / "tune"
    trials_dir = out_dir / "trials"

    job_args = {
        "model": args.model,
        "iterations": args.iterations,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "fraction": args.fraction,
        "device": args.device,
    }

    # 이어하기의 비교 가능성을 결정하는 값만 넣는다. device 는 속도만 바꾸고 결과를 바꾸지 않는다.
    signature = {
        "model": args.model,
        "dataset": str(dataset_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "fraction": args.fraction,
        "space": sorted(tune.SPACE),
    }

    started = time.time()
    poller: Poller | None = None
    try:
        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.is_file():
            raise RuntimeError(
                f"데이터셋 정의를 찾지 못했습니다: {data_yaml}. 데이터셋을 다시 등록하세요."
            )

        if args.restart:
            # 이어하기가 기본이라 "처음부터" 는 기록을 지우는 것으로만 표현된다.
            # 리포트를 **먼저** 지운다. 아래 삭제가 도중에 실패하면 기록은 사라졌는데 리포트만
            # 남아 시도 3개가 있는 것처럼 보인다. 어차피 사용자가 버리라고 한 것들이다.
            (out_dir / "tune.json").unlink(missing_ok=True)
            # **지웠는지 확인한다.** 파일 잠금 등으로 삭제가 조용히 실패하면 사용자는 새 탐색인
            # 줄 알지만 실제로는 옛 시도부터 이어가고, 그 기준으로 잰 개선폭이 화면에 나간다.
            for path in (tune_dir, trials_dir):
                shutil.rmtree(path, ignore_errors=True)
                if path.exists():
                    raise RuntimeError(
                        f"이전 탐색 기록을 지우지 못했습니다: {path}. "
                        "다른 프로그램이 이 폴더를 열고 있는지 확인한 뒤 다시 시도하세요."
                    )

        # 강제 종료가 append 도중에 떨어졌으면 마지막 줄이 잘려 있다. ultralytics 는 그 파일에서
        # 그대로 죽으므로 넘기기 전에 고친다(tune.repair_results 주석 참고).
        dropped = tune.repair_results(tune_dir)
        resumed = len(tune.read_results(tune_dir))
        if resumed:
            # 이어하기는 옛 시도와 새 시도를 한 표에 놓는다. 설정이 달라졌으면 그 표가 거짓이다 —
            # 기준(시도 1)은 옛 에폭·데이터 비율로 재고 새 시도는 다른 조건으로 재게 된다.
            # 조용히 섞느니 멈추고 "처음부터 다시" 를 고르게 한다.
            if args.iterations <= resumed:
                # ultralytics 는 start >= iterations 면 아무 시도도 하지 않고 끝난다.
                # 그러면 "3/2 완료" 같은 리포트가 남아 사용자가 무엇이 일어났는지 알 수 없다.
                raise RuntimeError(
                    f"이미 시도 {resumed}개가 끝나 있어 시도 횟수를 {args.iterations}로 "
                    "줄여서는 이어갈 수 없습니다. 더 큰 값을 넣거나 '처음부터 다시' 를 켜세요."
                )
            previous = signature_of(tune_dir)
            if not isinstance(previous, dict):
                raise RuntimeError(
                    f"이전 탐색 기록 {resumed}개가 어떤 설정으로 만들어졌는지 알 수 없어 "
                    "이어갈 수 없습니다. '처음부터 다시' 를 켜고 시작하세요."
                )
            if previous != signature:
                changed = sorted(
                    key for key in signature if previous.get(key) != signature.get(key)
                )
                raise RuntimeError(
                    f"이전 탐색과 설정이 달라 이어갈 수 없습니다({', '.join(changed)}). "
                    "같은 설정으로 맞추거나 '처음부터 다시' 를 켜고 시작하세요."
                )
        # 낡은 리포트는 **가드를 다 지난 뒤에** 지운다. 이 잡은 실행 중에도 리포트를 보여주므로
        # 다른 잡의 "completed 일 때만 읽는다" 가드를 쓸 수 없어 지워야 하는데, 위에서 거절된
        # 요청은 아무것도 바꾸지 않았으므로 지난 결과를 없앨 이유가 없다.
        (out_dir / "tune.json").unlink(missing_ok=True)
        trials_dir.mkdir(parents=True, exist_ok=True)

        # Tuner 도 같은 경로를 exist_ok 로 만든다(resume=True). 먼저 만들어 서명을 남긴다 —
        # 첫 시도가 실패해도 다음 실행이 무엇과 이어지는지 알아야 한다.
        tune_dir.mkdir(parents=True, exist_ok=True)
        (tune_dir / SIGNATURE_NAME).write_text(
            json.dumps(signature, ensure_ascii=False, default=str), encoding="utf-8"
        )

        write(
            events,
            {
                "t": "start",
                **job_args,
                "space": sorted(tune.SPACE),
                "resumed_from": resumed,
            },
        )
        if resumed:
            repaired = (
                f" (중단으로 깨진 기록 {dropped}줄은 버렸습니다)" if dropped else ""
            )
            write(
                events,
                {
                    "t": "progress",
                    "stage": "trial",
                    "message": f"이전 탐색의 시도 {resumed}개를 이어서 계속합니다.{repaired}",
                },
            )

        # 시도 subprocess(및 DDP 손자)가 sitecustomize 를 거치게 한다.
        os.environ["YOLOWEB_RUN_DIR"] = str(trials_dir)
        os.environ["YOLOWEB_NO_PREVIEW"] = "1"  # 시도끼리 미리보기를 덮어쓰는 것을 막는다
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(HOOKS_DIR), str(BACKEND_DIR), os.environ.get("PYTHONPATH", "")]
        ).strip(os.pathsep)

        poller = Poller(out_dir, tune_dir, events, job_args, resumed)
        poller.snapshot()  # 시작 직후에도 화면이 읽을 것이 있어야 한다
        poller.start()

        from ultralytics import YOLO

        YOLO(args.model).tune(
            data=str(data_yaml),
            space=tune.SPACE,
            iterations=args.iterations,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            fraction=args.fraction,
            device=args.device,
            project=str(out_dir),
            # tune_dir 를 <out_dir>/tune 으로 고정한다. 모듈 첫머리 3번 참고.
            resume=True,
            plots=False,
            save_period=-1,
        )

        # 마지막 시도를 데몬 스레드에 맡기지 않는다. 세워 놓고 이 스레드가 직접 마무리한다.
        poller.stop()
        poller.join(timeout=POLL_INTERVAL * 3)
        report = poller.snapshot()

        # 확인 시도 — 기준 하이퍼파라미터를 시드만 바꿔 한 번 더. 이것이 있어야 "이 상승이
        # 우연과 구분되는가" 를 이 설정에서 직접 답할 수 있다(measure_noise 주석 참고).
        baseline = report.get("baseline")
        if baseline and len(report.get("trials") or []) >= 2:
            measured = measure_noise(
                tune_dir,
                baseline,
                {
                    "model": args.model,
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "fraction": args.fraction,
                },
                str(data_yaml),
                args.device,
                announce=lambda n, total: write(
                    events,
                    {
                        "t": "progress",
                        "stage": "noise",
                        "message": f"확인 시도 {n}/{total} — 기준값을 다른 시드로 다시 돌려"
                        " 흔들림을 잽니다.",
                    },
                ),
            )
            if measured is not None:
                poller.noise = measured
            else:
                write(
                    events,
                    {
                        "t": "progress",
                        "stage": "noise",
                        "message": "확인 시도가 실패해 흔들림을 재지 못했습니다. "
                        "기본 기준으로만 판정합니다.",
                    },
                )
            report = poller.snapshot()

        write(
            events,
            {
                "t": "end",
                "status": "completed",
                "report": "tune.json",
                "iterations_done": report.get("iterations_done"),
                "gain": report.get("gain"),
                "elapsed_s": round(time.time() - started, 1),
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - 실패도 사용자에게 보여야 한다
        if poller is not None:
            poller.stop()
            poller.join(timeout=POLL_INTERVAL * 3)
            try:
                # 여기까지 끝난 시도는 남긴다. 몇 시간짜리 잡이라 통째로 버리면 안 된다.
                poller.snapshot()
            except Exception:  # noqa: BLE001
                pass
        write(
            events,
            {
                "t": "end",
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
