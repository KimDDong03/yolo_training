"""품질 검사가 지목한 이미지를 원본까지 지운다.

**되돌릴 수 없는 동작이다.** 그래서 이 모듈의 대부분은 지우는 코드가 아니라 지우지 않기로
판단하는 코드다. 지울 수 있는 경로는 네 조건을 전부 만족해야 한다.

1. 지금 `quality.json` 이 지목한 경로일 것 — 중복 묶음이나 누수 쌍에 실린 것.
   이게 폭발 반경을 "화면에 뜬 것" 으로 묶는다. 리포트는 이미 GROUPS_CAP/PAIRS_CAP 로
   잘려 있어서 허용 집합이 화면과 정확히 같다.
2. train.txt / val.txt 에 실려 있을 것 — 이 앱이 학습에 쓰는 사진일 것.
3. 등록된 root 안쪽일 것 — 이미지 서빙과 **같은** 경계(dataset_ingest.resolve_in_root).
4. 이미지 확장자일 것.

라벨도 지우지만 root 안쪽일 때만 지운다. `_label_for` 는 경로 전체에서 images 를 찾아
labels 로 바꾸므로, root 가 `.../dataset/images` 로 등록돼 있으면 라벨 경로가
`.../dataset/labels/...` 로 **root 밖으로 나간다.**

## 순서

파일을 먼저 지우고 **실제로 지워진 것만** 목록에서 뺀다. 못 지운 파일(윈도우 잠금)이
목록에 남아 있어야 다음 검사에 다시 떠서 재시도할 수 있다. 그 대신 파일은 지웠는데 목록을
못 고친 창이 생기는데 — 학습이 "없는 파일" 로 죽는 유일한 상태다 — `prune.pending.json`
을 먼저 적어 두고 기동 시 recover() 가 그 창을 닫는다.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from app.core import db, fsops
from app.core.config import DATASETS_DIR, IMAGE_SUFFIXES
from app.services import dataset_ingest, jobs

#: 삭제 중간에 죽었을 때 무엇을 지우던 중이었는지. recover() 가 읽는다.
PENDING_NAME = "prune.pending.json"
#: 삭제 원장. 되돌릴 수 없는 동작이라 무엇을 지웠는지는 반드시 남긴다.
LEDGER_NAME = "deleted.json"


class PruneError(Exception):
    """사용자에게 그대로 보여줄 수 있는 실패.

    status 를 함께 들고 다닌다. "지금은 안 된다"(409)와 "그건 못 지운다"(422)는
    사용자가 해야 할 일이 다르다 — 기다리는 것과 다시 고르는 것.
    """

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ 경로 다루기


def _key(path: str | Path) -> str:
    """경로 비교용 키. 윈도우의 대소문자·구분자 차이를 여기서 한 번에 흡수한다."""
    return os.path.normcase(str(Path(path).resolve()))


def _read_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def _write_list(path: Path, lines: list[str]) -> None:
    """임시 파일에 쓰고 갈아끼운다. 쓰는 도중에 죽어도 반쪽짜리 목록이 남지 않는다."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ 허용 집합


def _reported_paths(dataset_id: str) -> set[str]:
    """지금 품질 리포트가 지목한 이미지. 이 집합 밖은 지울 수 없다."""
    path = jobs.job_dir("quality", "dataset", dataset_id) / "quality.json"
    if not path.is_file():
        raise PruneError(
            "품질 검사 결과가 없습니다. 검사를 먼저 돌린 뒤 지울 사진을 고르세요."
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PruneError(f"품질 검사 결과를 읽지 못했습니다: {exc}") from exc

    out: set[str] = set()
    duplicates = report.get("duplicates") or {}
    if not duplicates.get("failed"):
        for group in duplicates.get("groups") or []:
            for image in group.get("images") or []:
                if image.get("path"):
                    out.add(_key(image["path"]))
    leakage = report.get("leakage") or {}
    if not leakage.get("failed"):
        for pair in leakage.get("pairs") or []:
            for side in ("train", "val"):
                if pair.get(side):
                    out.add(_key(pair[side]))
    return out


# ------------------------------------------------------------------ 가드


def _busy_reason(dataset_id: str) -> str | None:
    """지금 지우면 안 되는 이유. 없으면 None.

    run_manager.exclusive_delete 는 이 데이터셋이 **소유한** 잡만 본다. 학습 run 과
    run 소유인 오류 분석 잡은 그 그물에 걸리지 않는데, 둘 다 data.yaml 을 통해 이 목록을
    읽는다(analysis_worker.py 의 config["data"]). 도중에 파일이 사라지면 실패하거나
    반쪽짜리 결과를 낸다.
    """
    if db.query_one(
        "SELECT id FROM runs WHERE dataset_id = ? AND status IN ('queued','running')",
        (dataset_id,),
    ):
        return "이 데이터셋으로 학습이 진행 중이거나 대기 중입니다. 끝난 뒤에 지우세요."

    rows = db.query(
        "SELECT j.* FROM jobs j JOIN runs r ON r.id = j.owner_id"
        " WHERE j.owner_type = 'run' AND j.status = 'running' AND r.dataset_id = ?",
        (dataset_id,),
    )
    for row in rows:
        job = db.row_to_job(row)
        if job["kind"] == "analyze" and jobs.alive(job):
            return "이 데이터셋을 쓰는 오류 분석이 진행 중입니다. 끝난 뒤에 지우세요."
    return None


# ------------------------------------------------------------------ 원장·리포트


def _set_aside(path: Path) -> None:
    """깨진 파일을 지우지 않고 옆으로 치운다. 이름이 겹치면 지난 증거가 사라지므로
    시각을 붙여 매번 다른 이름을 쓴다."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.replace(path, path.with_name(f"{path.name}.corrupt.{stamp}"))


def _ledger_paths(dataset_dir: Path) -> set[str]:
    """원장에 이미 실린 경로. 복구가 같은 삭제를 두 번 적지 않게 한다."""
    path = dataset_dir / LEDGER_NAME
    if not path.is_file():
        return set()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(loaded, list):
        return set()
    return {_key(e["path"]) for e in loaded if isinstance(e, dict) and e.get("path")}


def _write_json(path: Path, payload: Any) -> None:
    """임시 파일에 쓰고 갈아끼운다. 쓰다가 죽어도 반쪽짜리 파일이 남지 않는다."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _append_ledger(dataset_dir: Path, records: list[dict[str, Any]]) -> str | None:
    """삭제 원장에 덧붙인다. 실패하면 사유를 돌려준다(예외를 내지 않는다).

    이 시점에는 파일이 이미 사라졌다. 여기서 예외를 내면 실제로 지워졌는데 "실패했다" 고
    답하게 된다 — 사용자가 다시 지우려 들고, 무엇이 사라졌는지도 알 수 없다.
    그래서 실패는 응답의 failed 목록으로 흘려보내 사실대로 말한다.
    """
    if not records:
        return None
    path = dataset_dir / LEDGER_NAME
    history: list[dict[str, Any]] = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded
            else:
                raise ValueError("원장이 배열이 아닙니다")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # 깨진 원장을 덮어쓰면 지난 삭제 기록까지 함께 사라진다. 옆으로 치워 남긴다.
            try:
                _set_aside(path)
            except OSError:
                return f"삭제 기록을 남기지 못했습니다(기존 원장이 깨졌습니다): {exc}"

    history.extend(records)
    try:
        _write_json(path, history)
    except OSError as exc:
        return f"파일은 지웠지만 삭제 기록({LEDGER_NAME})을 남기지 못했습니다: {exc}"
    return None


def _refresh_report(
    dataset: dict[str, Any],
    dataset_dir: Path,
    root: Path,
    train_count: int,
    val_count: int,
) -> dict[str, Any]:
    """검수 리포트와 DB report 를 다시 만든다.

    장수만 고치면 class_instances·issue_counts·box_stats 가 조용히 낡는다. 데이터셋 목록·
    검수 패널·새 학습 화면이 그 값을 그대로 그리므로 한 벌만 고치면 나머지가 거짓말을 한다.
    삭제는 드문 동작이라 전수 재스캔 비용을 감수한다.
    """
    report = dict(dataset.get("report") or {})
    report["train_count"] = train_count
    report["val_count"] = val_count

    try:
        info = dataset_ingest.scan(root)
    except dataset_ingest.IngestError:
        # 재스캔이 안 되면 장수만이라도 맞춘다. 통계가 낡는 것이 응답을 죽이는 것보다 낫다.
        info = None

    if info is not None:
        info["root"] = str(root)
        dataset_ingest.write_review(dataset_dir, info)
        names = dataset.get("classes") or []
        report.update(
            {
                "total_images": info["report"]["total_images"],
                "split_counts": info["report"]["split_counts"],
                "issue_counts": info["report"]["issue_counts"],
                "review_cap": info["report"]["review_cap"],
                "box_stats": info["box_stats"],
                "class_instances": {
                    names[cid] if cid < len(names) else f"class_{cid}": n
                    for cid, n in sorted(info["class_counts"].items())
                },
            }
        )

    db.execute(
        "UPDATE datasets SET report = ? WHERE id = ?",
        (json.dumps(report, ensure_ascii=False), dataset["id"]),
    )
    return report


# ------------------------------------------------------------------ 본체


def delete_images(dataset: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    """고른 이미지를 원본과 라벨까지 지우고 목록·리포트를 맞춘다."""
    from app.services import run_manager  # 순환 import 를 피한다

    dataset_id = str(dataset["id"])
    dataset_dir = DATASETS_DIR / dataset_id
    root = Path(str(dataset.get("root") or "")).resolve()

    if not paths:
        raise PruneError("지울 사진을 고르지 않았습니다.", 422)

    status = dataset_ingest.path_status(dataset)
    if not status["ok"]:
        raise PruneError(status["message"])

    with run_manager.exclusive_delete("dataset", dataset_id):
        blocked = _busy_reason(dataset_id)
        if blocked:
            raise PruneError(blocked)

        # 목록과 리포트를 **락 안에서** 읽는다. 밖에서 읽으면 삭제 요청 두 개가 겹칠 때
        # 나중 쪽이 낡은 목록을 그대로 다시 써서 먼저 지운 파일이 목록에 되살아난다 —
        # 파일은 없는데 목록에는 있는 상태, 학습이 죽는 그 상태다.
        allowed = _reported_paths(dataset_id)
        train = _read_list(dataset_dir / "train.txt")
        val = _read_list(dataset_dir / "val.txt")
        split_of = {_key(p): "train" for p in train}
        split_of.update({_key(p): "val" for p in val})

        targets: dict[str, Path] = {}
        for raw in paths:
            target = dataset_ingest.resolve_in_root(root, raw)
            if target is None:
                raise PruneError(
                    f"데이터셋 폴더 밖의 파일은 지울 수 없습니다: {raw}", 422
                )
            if target.suffix.lower() not in IMAGE_SUFFIXES:
                raise PruneError(f"이미지 파일이 아닙니다: {raw}", 422)
            key = _key(target)
            if key not in split_of:
                raise PruneError(
                    f"이미지 목록에 없는 파일은 지울 수 없습니다: {raw}", 422
                )
            if key not in allowed:
                raise PruneError(
                    f"품질 검사가 지목하지 않은 파일은 지울 수 없습니다: {raw}"
                    " — 검사를 다시 돌린 뒤 고르세요.",
                    422,
                )
            targets[key] = target

        # 한 쪽이 통째로 비면 학습이 그대로 죽는다. 지우기 전에 막는다.
        if not [p for p in train if _key(p) not in targets]:
            raise PruneError("학습용 사진이 한 장도 남지 않습니다. 일부만 고르세요.")
        if not [p for p in val if _key(p) not in targets]:
            raise PruneError("검증용 사진이 한 장도 남지 않습니다. 일부만 고르세요.")

        pending = dataset_dir / PENDING_NAME
        _write_json(
            pending,
            {"paths": [str(p) for p in targets.values()], "started_at": time.time()},
        )

        records: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []
        deleted: set[str] = set()

        for key, target in targets.items():
            record: dict[str, Any] = {
                "path": str(target),
                "split": split_of[key],
                "label": None,
                "deleted_at": time.time(),
                "ok": False,
                "error": None,
            }
            try:
                fsops.remove_file(target)
            except OSError as exc:
                record["error"] = str(exc)
                records.append(record)
                failed.append(
                    {"path": str(target), "error": f"파일을 지우지 못했습니다: {exc}"}
                )
                continue

            record["ok"] = True
            deleted.add(key)

            label = dataset_ingest._label_for(target)
            inside = dataset_ingest.resolve_in_root(root, label)
            if inside is None:
                if label.is_file():
                    failed.append(
                        {
                            "path": str(label),
                            "error": "라벨이 등록된 폴더 밖에 있어 지우지 않았습니다.",
                        }
                    )
            elif inside.is_file():
                try:
                    fsops.remove_file(inside)
                    record["label"] = str(inside)
                except OSError as exc:
                    failed.append(
                        {
                            "path": str(inside),
                            "error": f"라벨을 지우지 못했습니다: {exc}",
                        }
                    )
            records.append(record)

        # 실제로 지운 것만 목록에서 뺀다. 못 지운 파일은 다음 검사에 다시 떠서 재시도할 수 있다.
        kept_train = [p for p in train if _key(p) not in deleted]
        kept_val = [p for p in val if _key(p) not in deleted]
        _write_list(dataset_dir / "train.txt", kept_train)
        _write_list(dataset_dir / "val.txt", kept_val)

        ledger_error = _append_ledger(dataset_dir, records)
        if ledger_error:
            failed.append(
                {"path": str(dataset_dir / LEDGER_NAME), "error": ledger_error}
            )

        _refresh_report(dataset, dataset_dir, root, len(kept_train), len(kept_val))

        # 지운 사진이 그대로 실린 리포트를 계속 띄우면 화면이 거짓말을 한다.
        # 캐시(cache.npz)는 남긴다 — 재검사가 1초에 끝나는 근거다.
        report_path = jobs.job_dir("quality", "dataset", dataset_id) / "quality.json"
        try:
            fsops.remove_file(report_path)
        except OSError as exc:
            # 조용히 넘기면 화면이 지운 사진을 계속 보여준다. 사실대로 말한다.
            failed.append(
                {
                    "path": str(report_path),
                    "error": f"낡은 품질 리포트를 지우지 못했습니다. 다시 검사하세요: {exc}",
                }
            )

        # 뒷정리까지 끝난 뒤에 저널을 지운다. 중간에 죽으면 recover() 가 목록을 맞춘다.
        pending.unlink(missing_ok=True)

    return {
        "deleted": len(deleted),
        "failed": failed,
        "train_count": len(kept_train),
        "val_count": len(kept_val),
    }


def recover() -> None:
    """중단된 삭제를 마무리한다. 백엔드 기동 시 한 번 부른다.

    닫는 상태는 하나다 — **파일은 이미 사라졌는데 목록에는 남아 있는 것.** 학습이
    "없는 파일" 로 죽는 유일한 상태다. 아직 남아 있는 파일은 건드리지 않는다. 사용자가
    다시 고르면 되고, 여기서 대신 지우면 사용자가 취소한 삭제를 되살리게 된다.

    DB 의 report 통계는 여기서 고치지 않는다. 데이터셋마다 전수 재스캔을 하게 되어 기동이
    느려진다 — 장수는 다음 삭제나 재등록 때 맞춰진다.
    """
    if not DATASETS_DIR.is_dir():
        return

    for pending in DATASETS_DIR.glob(f"*/{PENDING_NAME}"):
        dataset_dir = pending.parent
        try:
            record = json.loads(pending.read_text(encoding="utf-8"))
            gone = {
                _key(p) for p in (record.get("paths") or []) if not Path(p).exists()
            }
        except (OSError, json.JSONDecodeError):
            # 저널은 _write_json 으로 갈아끼우므로 반쪽짜리가 남을 수 없다. 그래도 깨져
            # 있으면 무엇을 지우던 중이었는지가 유일한 단서이므로 지우지 말고 치워 둔다.
            try:
                _set_aside(pending)
            except OSError:
                pass
            continue

        if gone:
            for name in ("train.txt", "val.txt"):
                lines = _read_list(dataset_dir / name)
                kept = [p for p in lines if _key(p) not in gone]
                if len(kept) != len(lines):
                    _write_list(dataset_dir / name, kept)

            # 원장에 이미 실린 것은 다시 적지 않는다. 원장을 쓴 직후에 죽으면 다음 기동의
            # 복구가 같은 삭제를 한 번 더 적게 되고, 저널을 못 지우면 그것이 매번 반복된다.
            recorded = _ledger_paths(dataset_dir)
            fresh = [
                p
                for p in (record.get("paths") or [])
                if _key(p) in gone and _key(p) not in recorded
            ]
            error = _append_ledger(
                dataset_dir,
                [
                    {
                        "path": p,
                        "split": None,
                        "label": None,
                        "deleted_at": record.get("started_at"),
                        "ok": True,
                        "error": None,
                        "recovered": True,
                    }
                    for p in fresh
                ],
            )
            if error:
                # 기록을 못 남겼으면 저널을 지우지 않는다. 다음 기동이 다시 시도한다.
                continue
        pending.unlink(missing_ok=True)
