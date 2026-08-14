"""데이터셋 등록 — zip 업로드와 로컬 경로 지정 두 경로가 같은 파이프라인을 탄다.

zip 해제는 신뢰할 수 없는 외부 입력을 다루므로 safe_extract 의 방어를 절대 완화하지 말 것.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import yaml

from app.core.config import (
    ALLOWED_SUFFIXES,
    DATASETS_DIR,
    IMAGE_SUFFIXES,
    MAX_ENTRIES,
    MAX_UNCOMPRESSED_BYTES,
)

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}
# 이보다 크면 상세 리포트(빈 라벨 목록 등) 수집을 생략한다. 클래스 집계는 항상 전수로 한다.
FULL_VERIFY_LIMIT = 20_000


class IngestError(Exception):
    """사용자에게 그대로 보여줄 수 있는 등록 실패."""


# --------------------------------------------------------------------------- zip


def safe_extract(zip_path: Path, dest: Path) -> int:
    """zip 을 dest 아래로만 해제한다. 위험한 엔트리는 거부하고 그 외 확장자는 건너뛴다.

    Returns:
        실제로 해제한 파일 수.
    """
    dest.mkdir(parents=True, exist_ok=True)
    dest_root = dest.resolve()
    written = 0
    total_bytes = 0

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise IngestError("zip 파일을 열 수 없습니다. 손상되었거나 zip 형식이 아닙니다.") from exc

    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ENTRIES:
            raise IngestError(f"zip 안의 항목이 너무 많습니다 ({len(infos):,}개 > {MAX_ENTRIES:,}개).")

        for info in infos:
            name = info.filename
            if info.is_dir():
                continue

            # 심볼릭 링크 엔트리 거부 (상위 4비트가 파일 종류)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise IngestError(f"심볼릭 링크가 들어 있어 거부했습니다: {name}")

            pure = Path(name.replace("\\", "/"))
            if pure.is_absolute() or pure.drive or name.startswith("/") or ":" in name.split("/")[0]:
                raise IngestError(f"절대 경로 항목이 들어 있어 거부했습니다: {name}")
            if ".." in pure.parts:
                raise IngestError(f"상위 디렉터리로 나가는 항목이 들어 있어 거부했습니다: {name}")

            target = (dest_root / pure).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise IngestError(f"해제 폴더 밖을 가리키는 항목이 들어 있어 거부했습니다: {name}")

            if pure.suffix.lower() not in ALLOWED_SUFFIXES:
                continue

            total_bytes += info.file_size
            if total_bytes > MAX_UNCOMPRESSED_BYTES:
                raise IngestError("압축을 풀었을 때 용량이 상한을 넘습니다.")

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1

    if written == 0:
        raise IngestError("zip 안에서 사용할 수 있는 파일을 찾지 못했습니다 (이미지·라벨·yaml).")
    return written


# ------------------------------------------------------------------- 구조 감지


def _strip_wrapper(root: Path) -> Path:
    """단일 폴더로 한 겹 감싸인 zip 을 벗겨낸다."""
    current = root
    for _ in range(4):
        entries = [p for p in current.iterdir() if not p.name.startswith("__MACOSX")]
        if len(entries) == 1 and entries[0].is_dir():
            current = entries[0]
        else:
            break
    return current


def _find_yaml(root: Path) -> Path | None:
    candidates = [p for p in root.rglob("*.yaml")] + [p for p in root.rglob("*.yml")]
    named = [p for p in candidates if p.stem.lower() in {"data", "dataset"}]
    pool = named or candidates
    if not pool:
        return None
    return min(pool, key=lambda p: (len(p.relative_to(root).parts), p.name))


def _label_for(image: Path) -> Path:
    """이미지 경로에 대응하는 라벨 경로. images/ 를 labels/ 로 바꾸는 YOLO 관례를 따른다."""
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _split_of(image: Path, root: Path) -> str | None:
    try:
        rel = image.relative_to(root)
    except ValueError:
        return None
    for part in rel.parts:
        alias = SPLIT_ALIASES.get(part.lower())
        if alias:
            return alias
    return None


def _read_yaml_names(yaml_path: Path) -> list[str] | None:
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(n) for n in names]
    return None


def _read_classes_txt(root: Path) -> list[str] | None:
    for candidate in ("classes.txt", "obj.names", "names.txt"):
        hits = list(root.rglob(candidate))
        if hits:
            lines = [ln.strip() for ln in hits[0].read_text(encoding="utf-8").splitlines()]
            names = [ln for ln in lines if ln]
            if names:
                return names
    return None


def _parse_label(path: Path) -> tuple[list[int], list[str]]:
    """라벨 파일에서 클래스 인덱스 목록과 문제 메시지를 뽑는다."""
    class_ids: list[int] = []
    issues: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], [f"읽기 실패: {exc}"]

    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            issues.append(f"{lineno}행: 항목이 5개 미만")
            continue
        try:
            # int(float(...)) 로 파싱하면 "1.9" 가 조용히 클래스 1이 되고 "inf" 는 OverflowError 를 던진다.
            cid = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except (ValueError, OverflowError):
            issues.append(f"{lineno}행: 숫자로 해석할 수 없음")
            continue
        if cid < 0:
            issues.append(f"{lineno}행: 클래스 인덱스가 음수")
            continue
        # nan 은 모든 크기 비교가 거짓이라 범위 검사만으로는 통과해버린다.
        if not all(math.isfinite(v) for v in coords):
            issues.append(f"{lineno}행: 좌표에 nan/inf 가 있음")
            continue
        if any(v < -0.001 or v > 1.001 for v in coords):
            issues.append(f"{lineno}행: 좌표가 0~1 범위를 벗어남")
        class_ids.append(cid)
    return class_ids, issues


def scan(root: Path) -> dict[str, Any]:
    """이미지·라벨을 훑어 구조와 검수 리포트를 만든다."""
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise IngestError("이미지 파일을 찾지 못했습니다.")

    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": [], "unassigned": []}
    class_counts: dict[int, int] = {}
    missing_labels: list[str] = []
    empty_labels: list[str] = []
    label_issues: list[dict[str, Any]] = []
    max_class = -1

    # 라벨은 항상 전부 읽는다. 건너뛰면 클래스 개수를 알 수 없어 data.yaml 이 틀리게 생성된다
    # (yaml/classes.txt 가 없는 데이터셋은 라벨의 최대 인덱스가 유일한 근거다).
    # 큰 데이터셋에서는 문제 목록 수집만 상한을 둔다.
    detailed = len(images) <= FULL_VERIFY_LIMIT
    for image in images:
        splits[_split_of(image, root) or "unassigned"].append(image)

        label = _label_for(image)
        if not label.exists():
            if len(missing_labels) < 50:
                missing_labels.append(str(image.relative_to(root)))
            continue
        ids, issues = _parse_label(label)
        if not ids and not issues and detailed and len(empty_labels) < 50:
            empty_labels.append(str(image.relative_to(root)))
        for cid in ids:
            class_counts[cid] = class_counts.get(cid, 0) + 1
            max_class = max(max_class, cid)
        if issues and len(label_issues) < 50:
            label_issues.append({"file": str(label.relative_to(root)), "issues": issues[:5]})

    # 라벨만 있고 이미지가 없는 경우
    orphan_labels: list[str] = []
    image_stems = {p.with_suffix("").as_posix() for p in images}
    for label in root.rglob("*.txt"):
        if label.name.lower() in {"classes.txt", "names.txt", "obj.names", "train.txt", "val.txt"}:
            continue
        stem = label.with_suffix("").as_posix().replace("/labels/", "/images/")
        if stem not in image_stems and label.with_suffix("").as_posix() not in image_stems:
            if len(orphan_labels) < 50:
                orphan_labels.append(str(label.relative_to(root)))

    return {
        "images": images,
        "splits": splits,
        "class_counts": class_counts,
        "max_class": max_class,
        "detailed_report": detailed,
        "report": {
            "total_images": len(images),
            "split_counts": {k: len(v) for k, v in splits.items()},
            "missing_labels": missing_labels,
            "missing_label_shown": len(missing_labels),
            "empty_labels": empty_labels,
            "orphan_labels": orphan_labels,
            "label_issues": label_issues,
            "detailed_report": detailed,
        },
    }


# --------------------------------------------------------------------- 등록


def _write_list(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(str(p.resolve()) for p in images) + "\n", encoding="utf-8")


def build_dataset(
    dataset_dir: Path,
    root: Path,
    name: str,
    source: str,
    origin: str,
    val_ratio: float = 0.2,
    seed: int = 0,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """감지 → 분할 → data.yaml 생성 → 메타 반환. 원본(root)에는 아무것도 쓰지 않는다."""
    info = scan(root)
    splits = info["splits"]

    train = list(splits["train"])
    val = list(splits["val"])
    leftover = list(splits["unassigned"])

    auto_split = False
    if not train and not val:
        # 분할 정보가 전혀 없다 → 비율로 나눈다
        pool = leftover or list(info["images"])
        rng = random.Random(seed)
        rng.shuffle(pool)
        cut = max(1, int(len(pool) * (1 - val_ratio)))
        train, val = pool[:cut], pool[cut:]
        auto_split = True
    elif train and not val:
        rng = random.Random(seed)
        rng.shuffle(train)
        cut = max(1, int(len(train) * (1 - val_ratio)))
        train, val = train[:cut], train[cut:]
        auto_split = True
    elif leftover and train:
        train.extend(leftover)

    if not train or not val:
        raise IngestError("train/val 로 나눌 이미지가 부족합니다.")

    # 클래스 이름 결정
    names = class_names
    if not names:
        yaml_path = _find_yaml(root)
        if yaml_path:
            names = _read_yaml_names(yaml_path)
    if not names:
        names = _read_classes_txt(root)
    if not names:
        count = max(info["max_class"] + 1, 1)
        names = [f"class_{i}" for i in range(count)]
    if info["max_class"] >= len(names):
        names = names + [f"class_{i}" for i in range(len(names), info["max_class"] + 1)]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_list(dataset_dir / "train.txt", train)
    _write_list(dataset_dir / "val.txt", val)

    yaml_out = dataset_dir / "data.yaml"
    yaml_out.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir.resolve()),
                "train": "train.txt",
                "val": "val.txt",
                "names": {i: n for i, n in enumerate(names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = dict(info["report"])
    report.update(
        {
            "auto_split": auto_split,
            "val_ratio": val_ratio if auto_split else None,
            "train_count": len(train),
            "val_count": len(val),
            "class_instances": {names[cid] if cid < len(names) else f"class_{cid}": n
                                for cid, n in sorted(info["class_counts"].items())},
        }
    )

    meta = {
        "id": dataset_dir.name,
        "name": name,
        "source": source,
        "origin": origin,
        "root": str(root.resolve()),
        "yaml_path": str(yaml_out.resolve()),
        "classes": names,
        "report": report,
        "created_at": time.time(),
    }
    (dataset_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def ingest_zip(zip_path: Path, name: str, val_ratio: float = 0.2, seed: int = 0) -> dict[str, Any]:
    dataset_id = uuid.uuid4().hex[:12]
    dataset_dir = DATASETS_DIR / dataset_id
    data_dir = dataset_dir / "data"
    try:
        safe_extract(zip_path, data_dir)
        root = _strip_wrapper(data_dir)
        return build_dataset(
            dataset_dir, root, name, "zip", zip_path.name, val_ratio=val_ratio, seed=seed
        )
    except Exception:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise


def ingest_path(src: str, name: str, val_ratio: float = 0.2, seed: int = 0) -> dict[str, Any]:
    root = Path(src).expanduser()
    if not root.is_dir():
        raise IngestError(f"폴더를 찾을 수 없습니다: {src}")
    dataset_id = uuid.uuid4().hex[:12]
    dataset_dir = DATASETS_DIR / dataset_id
    try:
        return build_dataset(
            dataset_dir, root.resolve(), name, "path", str(root), val_ratio=val_ratio, seed=seed
        )
    except Exception:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise
