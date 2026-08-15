"""데이터 품질 판정 — 중복 · train/val 누수 · 클래스 불균형.

파일 I/O 도 모델 호출도 하지 않는 순수 계산만 둔다. 이미지를 읽고 특징을 뽑는 것은
quality_worker.py 가 하고, 여기는 그 배열만 받는다. 그래야 합성 배열로 판정을 시험할 수 있다.

## 임계값을 이 값으로 정한 근거 (실데이터 3개 전수 측정, .codex/phase-3.md)

눈으로 확인한 참양성(같은 사진, 파일명만 다름)은 NCC 0.9999~1.0000 / 코사인 0.9865~1.0000
에 모인다. brain-tumor 의 인접 MRI 슬라이스는 NCC 0.9957~0.9993 / 코사인 0.9816~0.9942 로
**코사인은 완전히 겹치고 NCC 만 갈린다.** 그래서 삭제 권고는 NCC 로 세운다.

두 팔을 OR 로 묶으면 brain-tumor 에서 확정 2,521쌍이 나왔다 — 그 데이터셋에 바이트동일
쌍은 0개다. AND 로 바꾸면서 잃은 참양성은 세 데이터셋에서 하나도 없었다.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

# ------------------------------------------------------------------ 임계값
# 한 벌만 둔다. 워커·테스트·next_actions 가 전부 여기서 가져다 쓴다 —
# 두 곳이 다른 선을 쓰면 서로를 부정한다.

#: dHash64 해밍 거리가 이 값 이하면 후보쌍. 넓게 잡고 계층 2 에서 거른다.
CANDIDATE_HAMMING = 8
#: 확정 조건 (둘 다 넘어야 한다). 누수 판정이 이 집합을 쓴다.
CONFIRM_COSINE = 0.98
CONFIRM_NCC = 0.99
#: 삭제를 권고할 수 있는 선. 인접 슬라이스처럼 "닮았지만 다른" 것을 여기서 잘라낸다.
DELETE_NCC = 0.9995
#: NCC 분모 보호 — 픽셀당 표준편차가 이보다 작으면 방향을 신뢰하지 않는다(0~255 스케일).
FLAT_STD = 1.0

#: 오염된 val 비율이 이 값을 넘으면 mAP 를 그대로 믿지 말라고 경고한다.
LEAK_RATIO_ALERT = 0.01
#: train 인스턴스가 이보다 적은 클래스는 학습이 안 될 수 있다고 알린다.
RARE_INSTANCES = 20

#: 화면에 싣는 최대 개수. total 은 언제나 정확히 센다.
GROUPS_CAP = 30
PAIRS_CAP = 30


# ------------------------------------------------------------------ 특징

def dhash_bits(gray9x8: np.ndarray) -> np.uint64:
    """9x8 그레이 배열 -> 가로 이웃 차분 64비트."""
    a = np.asarray(gray9x8, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view(np.uint64)[0]


def normalize_thumbs(thumbs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """32x32 썸네일을 NCC 용으로 정규화한다 (평균 제거 + 단위화).

    분산이 거의 없는 이미지는 방향이 잡음이라 0 벡터로 남긴다 — 그런 쌍은 NCC 가 0 이
    되어 확정되지 않는다. 0 으로 나누는 것을 막는 동시에 "확인 불가" 를 표현한다.
    """
    mat = np.asarray(thumbs, dtype=np.float32)
    if mat.size == 0:
        return mat.reshape(0, 0), np.zeros(0, dtype=bool)
    centered = mat - mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    flat = (norms.ravel() / np.sqrt(max(mat.shape[1], 1))) < FLAT_STD
    out = centered / np.maximum(norms, 1e-8)
    out[flat] = 0.0
    return out, flat


def center_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """데이터셋 평균을 뺀 뒤 단위화.

    원시 코사인은 쓸 수 없다 — post-ReLU GAP 가 비음수라 서로 무관한 이미지끼리도
    0.92~1.00 에 뭉친다. 평균을 빼야 방향이 갈린다.
    """
    mat = np.asarray(embeddings, dtype=np.float32)
    if mat.size == 0:
        return mat.reshape(0, 0)
    centered = mat - mat.mean(axis=0, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)


def candidate_pairs(hashes: np.ndarray, threshold: int = CANDIDATE_HAMMING):
    """해밍 거리가 threshold 이하인 (i, j, 거리). 전쌍 비교.

    N=20,000 전쌍이 1초라 상한이 필요 없다(실데이터 최대 2,689장).
    """
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.int64)
    dist = np.bitwise_count(h[:, None] ^ h[None, :])
    i, j = np.triu_indices(n, 1)
    keep = dist[i, j] <= threshold
    return i[keep], j[keep], dist[i, j][keep].astype(np.int64)


def exact_pairs(digests: Sequence[bytes]) -> set[tuple[int, int]]:
    """파일 바이트가 완전히 같은 쌍. 오탐이 있을 수 없는 유일한 계층이다."""
    buckets: dict[bytes, list[int]] = {}
    for idx, digest in enumerate(digests):
        buckets.setdefault(digest, []).append(idx)
    return {
        (a, b)
        for group in buckets.values()
        if len(group) > 1
        for k, a in enumerate(group)
        for b in group[k + 1 :]
    }


# ------------------------------------------------------------------ 그룹

def union_find(n: int, pairs: Iterable[tuple[int, int]]) -> dict[int, list[int]]:
    """확정쌍을 연결 요소로 묶는다. 반환은 {대표: 멤버 목록}, 크기 2 이상만."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    groups: dict[int, list[int]] = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    return {root: members for root, members in groups.items() if len(members) > 1}


def is_complete(members: Sequence[int], confirmed: set[tuple[int, int]]) -> bool:
    """모든 쌍이 직접 확정된 그룹인가.

    유사도는 전이적이지 않다. A~B 와 B~C 만 확정돼도 union-find 는 셋을 한 그룹으로
    묶는데, A 와 C 는 다른 사진일 수 있다. 그대로 "2장 지워도 된다" 고 말하면 사용자가
    멀쩡한 사진을 지운다. 그룹은 작으므로 모든 쌍을 실제로 확인한다.
    """
    for k, a in enumerate(members):
        for b in members[k + 1 :]:
            if (min(a, b), max(a, b)) not in confirmed:
                return False
    return True


# ------------------------------------------------------------------ 불균형

def imbalance(
    class_names: Sequence[str],
    train_counts: dict[int, int],
    val_counts: dict[int, int],
    train_images: dict[int, int],
    val_images: dict[int, int],
) -> dict[str, Any]:
    """split 별 클래스 분포. scan() 은 전역만 세므로(등록 핫패스) 여기서 분해한다."""
    rows: list[dict[str, Any]] = []
    for cls, name in enumerate(class_names):
        rows.append(
            {
                "cls": cls,
                "name": name,
                "train_instances": int(train_counts.get(cls, 0)),
                "val_instances": int(val_counts.get(cls, 0)),
                "train_images": int(train_images.get(cls, 0)),
                "val_images": int(val_images.get(cls, 0)),
            }
        )

    present = [r for r in rows if r["train_instances"] > 0]
    ratio = None
    if len(present) > 1:
        counts = [r["train_instances"] for r in present]
        ratio = round(max(counts) / max(min(counts), 1), 1)

    return {
        "classes": rows,
        "ratio": ratio,
        "missing_in_train": [r["name"] for r in rows if r["train_instances"] == 0],
        # val 에 정답이 없으면 그 클래스의 성능을 아예 알 수 없다.
        "missing_in_val": [
            r["name"] for r in rows if r["val_instances"] == 0 and r["train_instances"] > 0
        ],
        "rare_in_train": [
            r["name"] for r in rows if 0 < r["train_instances"] < RARE_INSTANCES
        ],
    }


# ------------------------------------------------------------------ 문장

def notes(scanned: int, unreadable: int, embedding_used: bool) -> list[str]:
    """이 검사가 무엇을 보지 않았는지. 한계를 먼저 말한다."""
    out = [
        "검사 대상은 train.txt 와 val.txt 에 실린 이미지뿐입니다."
        " test 로 나눈 이미지와 어느 목록에도 없는 이미지는 보지 않습니다.",
        "잘라낸 사진(크롭)은 찾지 못합니다. 같은 사진의 사본과 다시 저장한 사본을 찾습니다.",
    ]
    if unreadable:
        out.append(f"{unreadable}장은 열지 못해 검사에서 빠졌습니다.")
    if not embedding_used:
        out.append(
            "모델 특징을 쓰지 못해 밝기 패턴만으로 판정했습니다. 평소보다 놓치는 것이 많습니다."
        )
    out.append(f"{scanned:,}장을 서로 전부 비교했습니다.")
    return out


def duplicate_message(wasted: int, exact_groups: int, chain_groups: int) -> str:
    if wasted == 0 and chain_groups == 0:
        return "같은 사진이 중복으로 들어간 것은 없습니다."
    parts: list[str] = []
    if wasted:
        parts.append(
            f"{wasted:,}장을 지워도 학습에 쓰이는 사진은 그대로입니다"
            f" (완전히 같은 묶음 {exact_groups:,}개)."
        )
    if chain_groups:
        parts.append(
            f"서로 비슷한 사진이 이어진 묶음이 {chain_groups:,}개 있습니다."
            " 이쪽은 같은 사진이라고 단정할 수 없으니 눈으로 확인하세요."
        )
    return " ".join(parts)


def leakage_message(val_leaked: int, val_total: int, ratio: float) -> str:
    if val_leaked == 0:
        return "검증용 사진이 학습용에 섞여 들어간 것은 없습니다."
    line = (
        f"검증용 {val_total:,}장 가운데 {val_leaked:,}장({ratio * 100:.1f}%)이"
        " 학습용에도 들어 있습니다."
    )
    if ratio >= LEAK_RATIO_ALERT:
        return (
            line + " 모델이 이미 본 사진으로 채점하고 있어 지금 mAP 는 실제보다 높게"
            " 나옵니다. 겹치는 검증용 사진을 지우고 다시 학습·검증하세요."
        )
    return line + " 비율이 낮아 점수에 미치는 영향은 크지 않습니다."
