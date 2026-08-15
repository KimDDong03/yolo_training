"""검출 오류를 여섯 종류로 나누고, 각각을 고치면 mAP50 이 얼마나 오르는지 잰다.

mAP 한 숫자는 "무엇을 고쳐야 하는가" 를 말해 주지 않는다. 같은 mAP50 0.6 이라도 물체는
찾는데 박스가 헐거운 모델과 종류를 헷갈리는 모델은 처방이 정반대다. TIDE(Bolya et al. 2020)
는 오류를 유형별로 나눈 뒤 그 유형만 고쳤을 때의 mAP 상승분(dAP)을 재는데, 이 값이 곧
"먼저 손댈 것" 의 순서다.

tidecv 패키지는 반입하지 않는다(pycocotools 의존 + 2020년 이후 방치 + 출력이 그림이라
JSON 을 얻을 수 없음). 대신 ultralytics 가 이미 가진 ap_per_class 를 그대로 쓴다. 각 fix 는
IoU 재계산 없이 tp/conf/pred_cls/target_cls 배열 편집으로 표현되므로, 검증을 다시 돌리지
않고 ap_per_class 를 일곱 번(baseline 1 + 유형 6) 부르는 것으로 끝난다.

분류 순서와 IoU 마스크는 TIDE 원본(tidecv/quantify.py)을 그대로 따른다. 순서를 바꾸면
판정이 달라진다 — 예를 들어 "이미 잡힌 같은 클래스 정답과 0.6, 아직 안 잡힌 다른 클래스
정답과 0.9" 인 예측은 중복이 아니라 클래스 오류이고, 둘은 처방이 정반대다.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.services import diagnose

# 이만큼 겹치면 "그 물체를 찾은 것" (mAP50 과 같은 기준).
T_FG = 0.5
# 이보다 덜 겹치면 "아무것도 없는 곳을 짚은 것".
T_BG = 0.1
# 혼동 쌍을 셀 때 쓰는 신뢰도. 배포에서 실제로 보게 될 오류만 센다.
CONFUSION_CONF = 0.25
# 혼동 쌍을 몇 쌍까지 보여줄 것인가. 전부 나열하면 무엇이 문제인지 되레 안 보인다.
CONFUSION_CAP = 5

# 화면에 싣는 순서. 판정 순서(_classify_one)와는 다르다.
ERROR_KINDS = ("cls", "loc", "both", "dupe", "bkg", "miss")

# 예측 쪽 오류만 배열에 코드로 담는다. miss 는 정답 쪽 플래그다.
_CODES = {"cls": 0, "loc": 1, "both": 2, "dupe": 3, "bkg": 4}
_NO_ERROR = -1

LABELS = {
    "cls": "클래스 오류",
    "loc": "위치 오류",
    "both": "위치·클래스 둘 다",
    "dupe": "중복 검출",
    "bkg": "배경 오검출",
    "miss": "놓친 정답",
}

ADVICE = {
    "cls": "박스는 맞는데 종류를 틀립니다. 헷갈리는 두 클래스의 라벨 기준을 통일하거나, "
    "실제로 구분할 필요가 없다면 한 클래스로 합치는 편이 낫습니다.",
    "loc": "물체는 찾았는데 박스가 어긋납니다. imgsz 를 한 단계 올리고, 회전·스케일 증강이 "
    "강하면 낮추세요. 라벨 박스 자체가 헐겁게 그려져 있지 않은지도 확인하세요.",
    "both": "위치도 종류도 어긋났습니다. 특정 축을 손봐서 나아지는 종류가 아니라 학습이 "
    "부족한 것입니다. 에폭을 늘리고, 그래도 남으면 데이터를 늘리세요.",
    "dupe": "같은 물체를 두 번 잡습니다. 추론할 때 NMS IoU 를 낮추세요. 학습 라벨 자체가 "
    "겹쳐 있으면 모델이 겹치게 내도록 배우므로 그것도 확인하세요.",
    "bkg": "아무것도 없는 곳을 잡습니다. 배포 신뢰도를 올리면 대부분 사라집니다. 특정 배경에서 "
    "반복되면 그 장면의 라벨 없는 이미지를 학습 데이터에 넣으세요.",
    "miss": "물체를 아예 찾지 못합니다. 배포 신뢰도를 낮추면 즉시 더 잡히지만 오검출이 늘고, "
    "근본 대책은 놓친 사진과 비슷한 데이터를 늘리는 것입니다.",
}

NOTE = (
    "각 값은 그 오류만 고쳤을 때의 mAP50 상승분입니다. 오류끼리 겹치기 때문에 여섯 값을 "
    "더해도 남은 오차 전체가 되지는 않습니다."
)


def _round(value: Any, digits: int = 5) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _best(row: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """mask 가 참인 정답 중 IoU 가 가장 큰 것의 (IoU, index). 후보가 없으면 (0.0, -1)."""
    if not mask.any():
        return 0.0, -1
    index = int(np.where(mask, row, -1.0).argmax())
    return float(row[index]), index


def _classify_one(
    row: np.ndarray, unused_same: np.ndarray, unused_diff: np.ndarray, used_same: np.ndarray
) -> tuple[str, int]:
    """짝을 못 지은 예측 하나를 유형과 근거 정답으로 판정한다.

    순서가 곧 우선순위다(TIDE 원본과 같다). 위에서 먼저 걸리는 것이 이긴다.
    """
    iou, index = _best(row, unused_same)
    if T_BG <= iou <= T_FG:
        return "loc", index  # 맞는 종류인데 박스만 어긋났다

    iou, index = _best(row, unused_diff)
    if iou >= T_FG:
        return "cls", index  # 박스는 물체에 맞는데 종류를 틀렸다

    iou, index = _best(row, used_same)
    if iou >= T_FG:
        return "dupe", index  # 이미 다른 예측이 가져간 정답을 또 잡았다

    if row.max() <= T_BG:
        return "bkg", -1  # 어떤 정답과도 겹치지 않는다

    return "both", -1  # 위 어디에도 안 맞는 나머지 (원본의 OtherError)


def classify(records: list[dict[str, Any]], conf: float = 0.0) -> tuple[dict, dict]:
    """모든 이미지의 예측과 정답을 전역 배열로 펼치고, 각 예측에 오류 딱지를 붙인다.

    딕셔너리 리스트로 들면 검출 150만 개에서 수백 MB 가 되므로 병렬 numpy 배열로 담는다.

    dets: img cls conf tp err gt idx   (idx = 그 이미지 안에서 신뢰도순 정렬 후의 위치)
    gts:  img cls taken miss best_iou best_pred
    """
    d_img: list[int] = []
    d_cls: list[int] = []
    d_conf: list[float] = []
    d_tp: list[bool] = []
    d_err: list[int] = []
    d_gt: list[int] = []
    d_idx: list[int] = []
    g_img: list[int] = []
    g_cls: list[int] = []
    g_taken: list[bool] = []
    g_miss: list[bool] = []
    g_best_iou: list[float] = []
    g_best_pred: list[int] = []

    offset = 0
    for image, record in enumerate(records):
        matched = diagnose.match(record, conf)
        ious = matched["ious"]
        gt_cls = np.asarray(record["gt_cls"], dtype=int).ravel()
        p_cls = matched["p_cls"]
        p_conf = matched["p_conf"]
        p_hit = matched["p_hit"]
        taken = matched["gt_taken"] >= 0
        n_gt, n_pred = len(gt_cls), len(p_cls)

        covered = np.zeros(n_gt, dtype=bool)
        for pi in range(n_pred):
            if p_hit[pi]:
                kind, gt_index = None, -1
            elif n_gt == 0:
                kind, gt_index = "bkg", -1
            else:
                same = gt_cls == p_cls[pi]
                kind, gt_index = _classify_one(
                    ious[pi], same & ~taken, ~same & ~taken, same & taken
                )
                # 이 정답은 "설명된" 것이다. 고치면 잡히므로 놓침으로 세지 않는다.
                if kind in ("loc", "cls") and gt_index >= 0:
                    covered[gt_index] = True

            d_img.append(image)
            d_cls.append(int(p_cls[pi]))
            d_conf.append(float(p_conf[pi]))
            d_tp.append(bool(p_hit[pi]))
            d_err.append(_NO_ERROR if kind is None else _CODES[kind])
            d_gt.append(-1 if gt_index < 0 else offset + gt_index)
            d_idx.append(pi)

        # 놓침 = 아무도 못 잡았고, 위치·클래스 오류로도 설명되지 않은 정답.
        miss = ~taken & ~covered
        # 짝은 못 지었어도 "무언가 겹치기는 했는지" 는 라벨 오류 후보(2-4)가 묻는 질문이다.
        # 맞춘 예측까지 포함해서 재야 하므로 매칭 결과와는 다른 값이다.
        if n_gt and n_pred:
            best_iou = ious.max(axis=0)
            best_pred = ious.argmax(axis=0)
        else:
            best_iou = np.zeros(n_gt, dtype=np.float32)
            best_pred = np.full(n_gt, -1, dtype=int)

        for gi in range(n_gt):
            g_img.append(image)
            g_cls.append(int(gt_cls[gi]))
            g_taken.append(bool(taken[gi]))
            g_miss.append(bool(miss[gi]))
            g_best_iou.append(float(best_iou[gi]))
            g_best_pred.append(int(best_pred[gi]))
        offset += n_gt

    dets = {
        "img": np.asarray(d_img, dtype=np.int32),
        "cls": np.asarray(d_cls, dtype=np.int32),
        "conf": np.asarray(d_conf, dtype=np.float32),
        "tp": np.asarray(d_tp, dtype=bool),
        "err": np.asarray(d_err, dtype=np.int8),
        "gt": np.asarray(d_gt, dtype=np.int32),
        "idx": np.asarray(d_idx, dtype=np.int32),
    }
    gts = {
        "img": np.asarray(g_img, dtype=np.int32),
        "cls": np.asarray(g_cls, dtype=np.int32),
        "taken": np.asarray(g_taken, dtype=bool),
        "miss": np.asarray(g_miss, dtype=bool),
        "best_iou": np.asarray(g_best_iou, dtype=np.float32),
        "best_pred": np.asarray(g_best_pred, dtype=np.int32),
    }
    return dets, gts


def _evaluate(
    det_cls: np.ndarray, det_conf: np.ndarray, det_tp: np.ndarray, gt_cls: np.ndarray
) -> dict[int, float]:
    """{클래스: AP50}. ap_per_class 를 부르는 유일한 자리 — 여기가 일곱 번 돈다."""
    from ultralytics.utils.metrics import ap_per_class

    if len(gt_cls) == 0:
        # 정답이 하나도 없으면 평균낼 클래스가 없다. ap_per_class 는 여기서 nan 을 낸다.
        return {}
    result = ap_per_class(
        np.asarray(det_tp, dtype=bool).reshape(-1, 1),
        np.asarray(det_conf, dtype=np.float32),
        np.asarray(det_cls),
        np.asarray(gt_cls),
    )
    return {int(c): float(a) for c, a in zip(result[6], result[5][:, 0])}


def _fixed_arrays(dets: dict, gts: dict, kind: str) -> tuple:
    """오류 유형 하나만 고친 배열 사본. 원본은 건드리지 않는다."""
    if kind == "miss":
        # 놓친 정답을 없던 것으로 친다. 예측은 그대로다.
        return dets["cls"], dets["conf"], dets["tp"], gts["cls"][~gts["miss"]]

    if kind in ("both", "dupe", "bkg"):
        # 애초에 내지 말았어야 할 검출이므로 통째로 지운다 — 정밀도 분모에서도 빠져야 한다.
        keep = dets["err"] != _CODES[kind]
        return dets["cls"][keep], dets["conf"][keep], dets["tp"][keep], gts["cls"]

    det_cls = dets["cls"].copy()
    det_tp = dets["tp"].copy()
    rows = np.flatnonzero(dets["err"] == _CODES[kind])
    # 같은 정답을 여러 예측이 노릴 수 있다. 신뢰도가 높은 쪽이 가져간다.
    rows = rows[np.argsort(-dets["conf"][rows], kind="stable")]
    claimed = set(np.flatnonzero(gts["taken"]).tolist())
    for row in rows:
        target = int(dets["gt"][row])
        if target < 0:
            continue
        if kind == "cls":
            det_cls[row] = gts["cls"][target]
        if target in claimed:
            # 고쳐 봐야 중복이 된다. 여전히 오검출이므로 지우지 않고 남긴다 —
            # 지우면 정밀도를 깎지 않게 되어 상승분이 부풀려진다.
            det_tp[row] = False
        else:
            claimed.add(target)
            det_tp[row] = True
    return det_cls, dets["conf"], det_tp, gts["cls"]


def _counts(dets: dict, gts: dict) -> dict[str, int]:
    counts = {kind: int((dets["err"] == code).sum()) for kind, code in _CODES.items()}
    counts["miss"] = int(gts["miss"].sum())
    return counts


def _per_class_counts(dets: dict, gts: dict, names: dict[int, str]) -> list[dict[str, Any]]:
    """클래스별 오류 건수.

    예측 쪽 오류는 모델이 말한 클래스로, 놓침은 정답 클래스로 센다.
    """
    rows = []
    for cls_index in sorted(names):
        # 건수는 따로 담는다 — 오류 유형 하나가 "cls" 라서 클래스 번호와 키가 부딪힌다.
        counts = {
            kind: int(((dets["err"] == code) & (dets["cls"] == cls_index)).sum())
            for kind, code in _CODES.items()
        }
        counts["miss"] = int((gts["miss"] & (gts["cls"] == cls_index)).sum())
        rows.append({"cls": int(cls_index), "name": names[cls_index], "counts": counts})
    return rows


def _confusion_pairs(dets: dict, gts: dict, names: dict[int, str]) -> list[dict[str, Any]]:
    """어떤 종류를 어떤 종류로 잘못 부르는가. 배포 임계값 이상만 센다."""
    selected = np.flatnonzero(
        (dets["err"] == _CODES["cls"]) & (dets["conf"] >= CONFUSION_CONF) & (dets["gt"] >= 0)
    )
    tally: dict[tuple[int, int], int] = {}
    for row in selected:
        key = (int(dets["cls"][row]), int(gts["cls"][int(dets["gt"][row])]))
        tally[key] = tally.get(key, 0) + 1
    ordered = sorted(tally.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "pred": names.get(pred, str(pred)),
            "gt": names.get(truth, str(truth)),
            "count": count,
        }
        for (pred, truth), count in ordered[:CONFUSION_CAP]
    ]


def error_breakdown(
    records: list[dict[str, Any]], names: dict[int, str], collection_conf: float = 0.001
) -> dict[str, Any]:
    """report.json 의 "tide" 에 그대로 들어가는 값.

    collection_conf 는 검증이 예측을 모은 하한이다(계산에는 쓰지 않고 기록만 한다).
    분해 자체는 모아 온 예측 전부를 쓴다 — 신뢰도로 걸러 내면 배경 오검출과 중복이
    사라져 정작 재고 싶은 것이 안 보인다.
    """
    dets, gts = classify(records, 0.0)
    baseline = _evaluate(dets["cls"], dets["conf"], dets["tp"], gts["cls"])
    counts = _counts(dets, gts)

    errors = []
    for kind in ERROR_KINDS:
        fixed = _evaluate(*_fixed_arrays(dets, gts, kind))
        # 놓침을 고치면 정답이 하나도 안 남는 클래스가 생길 수 있다. 그 클래스가 평균에서
        # 빠지면 분모가 줄어 상승분이 저절로 부풀려지므로, 양쪽 평균을 같은 클래스 집합으로 낸다.
        common = [c for c in baseline if c in fixed]
        dropped = sorted(c for c in baseline if c not in fixed)
        if common:
            delta = float(np.mean([fixed[c] for c in common])) - float(
                np.mean([baseline[c] for c in common])
            )
        else:
            delta = 0.0
        naive = (float(np.mean(list(fixed.values()))) if fixed else 0.0) - (
            float(np.mean(list(baseline.values()))) if baseline else 0.0
        )
        errors.append(
            {
                "kind": kind,
                "label": LABELS[kind],
                "count": counts[kind],
                "dap": _round(delta),
                "dap_naive": _round(naive),
                "dropped_classes": dropped,
                "advice": ADVICE[kind],
            }
        )

    total = sum(e["dap"] for e in errors if (e["dap"] or 0) > 0)
    for error in errors:
        error["share"] = _round((error["dap"] or 0) / total, 4) if total > 0 else None

    return {
        "baseline_map50": _round(np.mean(list(baseline.values())) if baseline else 0.0),
        "baseline_classes": sorted(baseline),
        "params": {
            "collection_conf": collection_conf,
            "t_fg": T_FG,
            "t_bg": T_BG,
            "metric": "mAP50",
        },
        "errors": errors,
        "per_class_counts": _per_class_counts(dets, gts, names),
        "confusion_pairs": _confusion_pairs(dets, gts, names),
        "note": NOTE,
    }
