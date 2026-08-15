"""데이터 품질 검사 워커 — 중복 · train/val 누수 · 클래스 불균형.

학습 워커·분석 워커와 같은 규약을 따른다: 독립 프로세스, jsonl 로 진행 상황 append.

**원본 폴더 경로(root)를 인자로 받지 않는다.** 이미지 목록의 진실은 train.txt / val.txt 이고
둘 다 데이터셋 폴더 안에 절대 경로로 들어 있다. root 를 따로 받으면 DB 의 root 와 목록이
어긋난 상태를 워커가 다시 만들게 된다(그 상태는 실제로 발생한다 — dataset_ingest.path_status).
그래서 test 로 나눈 이미지와 어느 목록에도 없는 이미지는 이 검사의 대상이 아니다.

파일은 이미지당 한 번만 읽는다. 진짜 병목은 JPEG 디코딩이라, 해시·썸네일·임베딩이
같은 디코딩 결과를 나눠 쓴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import apply_offline_env  # noqa: E402

apply_offline_env()

from app.services import quality  # noqa: E402  (offline 환경을 잡은 뒤에 가져온다)

BATCH = 64
THUMB = 32
EMBED_SIDE = 256


def write(path: Path, payload: dict) -> None:
    payload.setdefault("ts", time.time())
    # 한 줄을 한 번의 write 로 내보내야 리더가 잘린 JSON 을 보지 않는다.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def read_list(path: Path) -> list[Path]:
    if not path.is_file():
        return []
    return [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def decode(path: Path):
    """한 번 열어 축소 배열을 돌려준다. JPEG 는 DCT 단계에서 줄여 디코딩을 아낀다."""
    from PIL import Image

    image = Image.open(path)
    image.draft("RGB", (EMBED_SIDE, EMBED_SIDE))
    image = image.convert("RGB")
    image.thumbnail((EMBED_SIDE, EMBED_SIDE), Image.Resampling.BILINEAR)
    return image


def features(image):
    """축소 이미지 하나에서 dHash 용 9x8 과 NCC 용 32x32 를 뽑는다."""
    from PIL import Image

    gray = image.convert("L")
    small = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    thumb = np.asarray(
        gray.resize((THUMB, THUMB), Image.Resampling.BILINEAR), dtype=np.float32
    )
    return small, thumb.ravel()


def label_counts(paths, splits, class_count):
    """split 별 클래스 인스턴스 수와 등장 이미지 수. 라벨 파싱은 등록 코드를 그대로 쓴다."""
    from app.services import dataset_ingest

    inst = {"train": {}, "val": {}}
    imgs = {"train": {}, "val": {}}
    for path, split in zip(paths, splits):
        label = dataset_ingest._label_for(path)
        if not label.exists():
            continue
        ids, _, _ = dataset_ingest._parse_label(label)
        seen = set()
        for cid in ids:
            if not 0 <= cid < class_count:
                continue
            inst[split][cid] = inst[split].get(cid, 0) + 1
            seen.add(cid)
        for cid in seen:
            imgs[split][cid] = imgs[split].get(cid, 0) + 1
    return inst, imgs


def class_names(dataset_dir: Path) -> list[str]:
    from app.services import dataset_ingest

    names = dataset_ingest._read_yaml_names(dataset_dir / "data.yaml")
    return names or []


# ------------------------------------------------------------------ 캐시

def load_cache(directory: Path, imgsz: int):
    """(경로, mtime, size) 가 그대로인 항목만 재사용한다. 어긋나면 그 항목만 다시 계산한다.

    **imgsz 가 다르면 통째로 버린다.** 임베딩은 입력 해상도에 따라 달라진다(같은 이미지의
    224 임베딩과 640 임베딩의 중심화 코사인이 0.64 였다). 캐시를 가로질러 재사용하면
    리포트에는 640 이라고 적으면서 224 로 만든 값으로 판정하게 된다 — 조용히 틀린다.
    """
    index_path, blob_path = directory / "cache.json", directory / "cache.npz"
    if not index_path.is_file() or not blob_path.is_file():
        return {}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if int(index.get("imgsz", -1)) != int(imgsz):
            return {}
        blob = np.load(blob_path)
        keys = index["keys"]
        hashes, thumbs, embeds = blob["hashes"], blob["thumbs"], blob["embeds"]
        digests = index["digests"]
        if not (len(keys) == len(hashes) == len(thumbs) == len(digests)):
            return {}
        has_embed = len(embeds) == len(keys)
        return {
            tuple(k): (
                np.uint64(hashes[i]),
                thumbs[i],
                embeds[i] if has_embed else None,
                bytes.fromhex(digests[i]),
            )
            for i, k in enumerate(keys)
        }
    except Exception:  # noqa: BLE001 - 캐시가 깨졌으면 그냥 전량 재계산한다
        return {}


def save_cache(directory: Path, imgsz: int, keys, hashes, thumbs, embeds, digests) -> None:
    try:
        np.savez_compressed(
            directory / "cache.npz",
            hashes=np.asarray(hashes, dtype=np.uint64),
            thumbs=np.asarray(thumbs, dtype=np.float32),
            embeds=(np.asarray(embeds, dtype=np.float32) if embeds is not None
                    else np.zeros((0, 0), dtype=np.float32)),
        )
        (directory / "cache.json").write_text(
            json.dumps({"imgsz": int(imgsz),
                        "keys": [list(k) for k in keys],
                        "digests": [d.hex() for d in digests]}),
            encoding="utf-8",
        )
    except OSError:
        pass  # 캐시는 있으면 좋은 것이지 결과의 일부가 아니다


# ------------------------------------------------------------------ 섹션

def section(label: str, build):
    """한 섹션이 깨져도 나머지 리포트는 살린다.

    조용히 키를 빼면 "예전 리포트라 없는 것" 과 구분되지 않는다. 실패를 값으로 남긴다.
    """
    try:
        return build()
    except Exception as exc:  # noqa: BLE001
        return {"failed": True, "message": f"{label}를 계산하지 못했습니다: {exc}"}


def build_duplicates(scanned, confirmed, deletable, exact_set, paths, splits):
    """확정쌍을 묶어 "지워도 되는 것" 과 "눈으로 확인할 것" 으로 나눈다.

    그룹은 확정쌍 전체로 묶는다. 지워도 되는 것만 묶으면 "닮았지만 못 지우는" 묶음
    (같은 환자의 인접 슬라이스 같은 것)이 화면에서 통째로 사라진다 — 지울 대상은
    아니어도 사용자가 알아야 하는 사실이다.
    """
    rendered, wasted, exact_groups, review_groups, dup_images = [], 0, 0, 0, 0
    for members in quality.union_find(scanned, confirmed).values():
        members = sorted(members)
        pairs_of = [(a, b) for k, a in enumerate(members) for b in members[k + 1:]]
        complete = quality.is_complete(members, confirmed)
        all_deletable = complete and all(p in deletable for p in pairs_of)
        all_exact = all_deletable and all(p in exact_set for p in pairs_of)
        if all_exact:
            kind = "exact"
        elif all_deletable:
            kind = "near"
        elif complete:
            kind = "similar"   # 확정은 됐지만 지우라고 말할 만큼 같지는 않다
        else:
            kind = "chain"     # 일부 쌍만 확정 — 유사도는 전이적이지 않다
        if all_deletable:
            wasted += len(members) - 1
            exact_groups += 1 if all_exact else 0
        else:
            review_groups += 1
        dup_images += len(members)
        rendered.append({
            "size": len(members),
            "kind": kind,
            "images": [{"path": str(paths[m]), "split": splits[m]} for m in members],
        })

    # 지울 수 있는 것 먼저, 그다음 큰 묶음 먼저.
    order = {"exact": 0, "near": 1, "similar": 2, "chain": 3}
    rendered.sort(key=lambda g: (order[g["kind"]], -g["size"]))
    return {
        "wasted": wasted,
        "image_total": dup_images,
        "group_total": len(rendered),
        "groups_cap": quality.GROUPS_CAP,
        "groups": rendered[: quality.GROUPS_CAP],
        "message": quality.duplicate_message(wasted, exact_groups, review_groups),
    }


def build_leakage(i, j, ham, cos, ncc, confirmed_mask, is_exact, split_arr,
                  paths, splits, has_embed):
    """train 과 val 을 가로지르는 확정쌍. 오염된 val 은 쌍이 아니라 **이미지**로 센다."""
    cross = confirmed_mask & (split_arr[i] != split_arr[j])
    leaked: set[int] = set()
    pairs = []
    for k in np.flatnonzero(cross):
        a, b = int(i[k]), int(j[k])
        train_idx, val_idx = (a, b) if splits[a] == "train" else (b, a)
        leaked.add(val_idx)
        pairs.append({
            "train": str(paths[train_idx]),
            "val": str(paths[val_idx]),
            "hamming": int(ham[k]),
            "cosine": round(float(cos[k]), 4) if has_embed else None,
            "ncc": round(float(ncc[k]), 4),
            "exact": bool(is_exact[k]),
        })
    pairs.sort(key=lambda p: (not p["exact"], -p["ncc"]))
    val_total = int((split_arr == "val").sum())
    ratio = round(len(leaked) / val_total, 4) if val_total else 0.0
    return {
        "val_leaked": len(leaked),
        "val_total": val_total,
        "ratio": ratio,
        "exact_pairs": int(sum(1 for p in pairs if p["exact"])),
        "pair_total": len(pairs),
        "pairs_cap": quality.PAIRS_CAP,
        "pairs": pairs[: quality.PAIRS_CAP],
        "message": quality.leakage_message(len(leaked), val_total, ratio),
    }


def build_imbalance(dataset_dir: Path, paths, splits):
    names = class_names(dataset_dir)
    if not names:
        return {"failed": True,
                "message": "data.yaml 에서 클래스 이름을 읽지 못해 분포를 세지 못했습니다."}
    inst, imgs = label_counts(paths, splits, len(names))
    return quality.imbalance(names, inst["train"], inst["val"], imgs["train"], imgs["val"])


# ------------------------------------------------------------------ 본체

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events = Path(args.events).resolve()

    write(events, {"t": "start", "imgsz": args.imgsz, "device": args.device})
    started = time.time()

    try:
        train = read_list(dataset_dir / "train.txt")
        val = read_list(dataset_dir / "val.txt")
        if not train and not val:
            raise RuntimeError(
                "이미지 목록(train.txt / val.txt)이 비어 있습니다. 데이터셋을 다시 등록하세요."
            )

        paths = train + val
        splits = ["train"] * len(train) + ["val"] * len(val)

        write(events, {"t": "progress", "stage": "scan",
                       "message": f"{len(paths):,}장을 읽는 중"})

        cache = load_cache(out_dir, args.imgsz)
        # 앞선 실행이 임베딩을 못 만들었으면 캐시에도 없다. 이번에는 만들 수 있으므로
        # "임베딩이 빠진 캐시" 는 재사용하지 않는다.
        keep_paths, keep_splits = [], []
        keys, hashes, thumbs, digests, embeds = [], [], [], [], []
        pending, pending_idx = [], []
        unreadable = 0
        model = None
        embed_error: str | None = None

        def flush(force: bool = False):
            """모아 둔 배치의 임베딩을 채운다. 모델을 못 쓰면 조용히 비워 둔다."""
            nonlocal model, embed_error, pending, pending_idx
            if not pending or (len(pending) < BATCH and not force):
                return
            if embed_error is None and model is None:
                try:
                    from app.core.config import WEIGHTS_DIR
                    from ultralytics import YOLO

                    weights = WEIGHTS_DIR / "yolo11n.pt"
                    if not weights.is_file():
                        raise FileNotFoundError(f"가중치를 찾지 못했습니다: {weights}")
                    model = YOLO(str(weights))
                except Exception as exc:  # noqa: BLE001
                    embed_error = f"{type(exc).__name__}: {exc}"
            if embed_error is None and model is not None:
                try:
                    out = model.embed(
                        [a[:, :, ::-1] for a in pending],
                        imgsz=args.imgsz, device=args.device, verbose=False,
                    )
                    for k, tensor in zip(pending_idx, out):
                        embeds[k] = tensor.detach().cpu().numpy().astype(np.float32)
                except Exception as exc:  # noqa: BLE001
                    embed_error = f"{type(exc).__name__}: {exc}"
            pending, pending_idx = [], []

        for n, (path, split) in enumerate(zip(paths, splits)):
            try:
                stat = path.stat()
                key = (str(path), int(stat.st_mtime), int(stat.st_size))
            except OSError:
                unreadable += 1
                continue

            hit = cache.get(key)
            if hit is not None and hit[2] is not None:
                keep_paths.append(path); keep_splits.append(split)
                keys.append(key); hashes.append(hit[0]); thumbs.append(hit[1])
                embeds.append(hit[2]); digests.append(hit[3])
            else:
                try:
                    image = decode(path)
                    small, thumb = features(image)
                    digest = hashlib.blake2b(path.read_bytes(), digest_size=16).digest()
                except Exception:  # noqa: BLE001 - 깨진 파일 한 장이 검사를 죽이면 안 된다
                    unreadable += 1
                    continue
                keep_paths.append(path); keep_splits.append(split)
                keys.append(key)
                hashes.append(quality.dhash_bits(small))
                thumbs.append(thumb)
                digests.append(digest)
                embeds.append(None)
                pending.append(np.asarray(image)); pending_idx.append(len(embeds) - 1)
                flush()

            if n and n % 500 == 0:
                write(events, {"t": "progress", "stage": "scan",
                               "message": f"{n:,}/{len(paths):,}장"})
        flush(force=True)

        scanned = len(keep_paths)
        if scanned < 2:
            raise RuntimeError(f"비교할 이미지가 부족합니다 ({scanned}장).")

        write(events, {"t": "progress", "stage": "compare",
                       "message": f"{scanned:,}장을 서로 비교하는 중"})

        H = np.asarray(hashes, dtype=np.uint64)
        T, _ = quality.normalize_thumbs(np.stack(thumbs))
        split_arr = np.asarray(keep_splits)

        use_embed = embed_error is None and all(e is not None for e in embeds)
        if use_embed:
            E = quality.center_embeddings(np.stack(embeds))
        else:
            E = None

        i, j, ham = quality.candidate_pairs(H)
        exact_set = quality.exact_pairs(digests)
        is_exact = np.array(
            [(int(a), int(b)) in exact_set for a, b in zip(i, j)], dtype=bool
        ) if len(i) else np.zeros(0, dtype=bool)

        ncc = (np.einsum("ij,ij->i", T[i], T[j]) if len(i) else np.zeros(0, dtype=np.float32))
        if E is not None and len(i):
            cos = np.einsum("ij,ij->i", E[i], E[j])
        else:
            # 임베딩이 없으면 코사인 조건을 통과시키고 NCC 만으로 판정한다.
            # 놓치는 것이 늘지만 없는 근거를 지어내지는 않는다.
            cos = np.full(len(i), 1.0, dtype=np.float32)

        confirmed_mask = is_exact | (
            (cos >= quality.CONFIRM_COSINE) & (ncc >= quality.CONFIRM_NCC)
        )
        deletable_mask = is_exact | (
            (cos >= quality.CONFIRM_COSINE) & (ncc >= quality.DELETE_NCC)
        )

        confirmed = {(int(a), int(b)) for a, b in zip(i[confirmed_mask], j[confirmed_mask])}
        deletable = {(int(a), int(b)) for a, b in zip(i[deletable_mask], j[deletable_mask])}

        # 세 섹션은 서로 독립적으로 실패한다. 한 섹션이 깨졌다고 나머지 결과까지
        # 사라지면 안 되고, 조용히 빠지면 "예전 리포트라 없는 것" 과 구분되지 않는다.
        duplicates = section(
            "중복 검사",
            lambda: build_duplicates(
                scanned, confirmed, deletable, exact_set, keep_paths, keep_splits
            ),
        )
        leakage = section(
            "누수 검사",
            lambda: build_leakage(
                i, j, ham, cos, ncc, confirmed_mask, is_exact, split_arr,
                keep_paths, keep_splits, E is not None,
            ),
        )
        write(events, {"t": "progress", "stage": "labels", "message": "라벨을 세는 중"})
        balance = section(
            "클래스 분포", lambda: build_imbalance(dataset_dir, keep_paths, keep_splits)
        )

        report = {
            "schema_version": 1,
            "dataset_id": dataset_dir.name,
            "created_at": time.time(),
            "elapsed_s": round(time.time() - started, 1),
            "params": {
                "imgsz": args.imgsz,
                "device": args.device,
                "hamming": quality.CANDIDATE_HAMMING,
                "cosine": quality.CONFIRM_COSINE,
                "ncc": quality.CONFIRM_NCC,
                "delete_ncc": quality.DELETE_NCC,
                "embedding": True if use_embed else {
                    "used": False,
                    "reason": embed_error or "임베딩을 만들지 못했습니다.",
                },
            },
            "counts": {
                "train": int((split_arr == "train").sum()),
                "val": int((split_arr == "val").sum()),
                "scanned": scanned,
                "unreadable": unreadable,
                "candidate_pairs": int(len(i)),
            },
            "duplicates": duplicates,
            "leakage": leakage,
            "imbalance": balance,
            "notes": quality.notes(scanned, unreadable, use_embed),
        }
        (out_dir / "quality.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
        save_cache(out_dir, args.imgsz, keys, hashes, thumbs,
                   embeds if use_embed else None, digests)
        write(events, {"t": "end", "status": "completed", "report": "quality.json",
                       "elapsed_s": round(time.time() - started, 1)})
        return 0
    except Exception as exc:  # noqa: BLE001 - 실패도 사용자에게 보여야 한다
        write(events, {"t": "end", "status": "failed", "error": str(exc),
                       "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
