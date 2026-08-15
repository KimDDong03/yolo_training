"""모델이 틀린 게 아니라 라벨이 틀렸을 법한 자리를 골라낸다.

성능이 안 오를 때 사람들은 먼저 파라미터를 만진다. 그런데 라벨이 틀려 있으면 무엇을 바꿔도
틀린 것을 더 정확히 배울 뿐이다. 모델과 라벨이 어긋나는 자리를 사진으로 보여 주면
비전문가도 어느 쪽이 틀렸는지 눈으로 판단할 수 있다.

새 추론은 하지 않는다. 오류 분해(tide.py)가 이미 붙여 둔 딱지를 다시 읽을 뿐이다.

여기서 가장 중요한 것은 **오탐을 내지 않는 것**이다. 아니라고 판명되는 후보를 몇 개만
보여 줘도 사용자는 이 목록 전체를 믿지 않게 되고, 그러면 진짜 라벨 오류도 함께 묻힌다.
그래서 판정은 두 겹으로 막는다 — 모델이 못 믿을 상태면 모델 근거를 쓰는 신호를 통째로 끄고,
켜져 있을 때도 "그 클래스에 한해 모델이 실제로 맞히고 있다" 는 증거를 요구한다.

한계는 화면에 반드시 적는다. 이건 검증(val) 셋만 본 결과이고, 어긋났다는 사실이 곧
라벨이 틀렸다는 뜻은 아니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.services import diagnose, tide

# "모델이 확신했다" 로 볼 선.
HIGH_CONF = 0.70
# 이미지 안에 뒷받침할 근거가 없을 때 요구하는 선. 여기를 낮추면 배경 오검출이 쏟아진다.
VERY_HIGH_CONF = 0.90
# 이 정도 겹치면 "같은 물체를 가리키고 있다" 고 말할 수 있다. TIDE 의 0.5 는 옆 물체를
# 잘못 집은 경우와 구분되지 않아 라벨 오기 판정에 그대로 쓰면 안 된다.
TIGHT_IOU = 0.75
# 정답 박스끼리 이만큼 겹치면 중복으로 본다. 낮추면 군중·정체 사진에서 오탐이 난다.
DUP_GT_IOU = 0.80
# 그 자리에 물체 비슷한 것조차 없다고 말할 수 있는 선 (오류 분해의 배경 기준과 같다).
NO_SIGNAL_IOU = 0.10
# 이 클래스에 대한 모델 판단을 근거로 쓸 수 있는 최소치.
MIN_CLASS_PRECISION = 0.50
MIN_CLASS_RECALL = 0.50
MIN_CLASS_INSTANCES = 20
# 모델 전체가 이보다 나쁘면 모델 근거를 아예 쓰지 않는다.
MIN_MODEL_MAP50 = 0.30
# 작은 물체는 모델이 원래 못 찾는다. 그걸 라벨 오류라고 부르면 안 된다.
MIN_BOX_AREA = 0.002
# 한 종류가 목록을 독차지하면 다른 신호가 안 보인다.
ISSUE_KIND_CAP = 10
# 갤러리(60장)보다 훨씬 적다. 이건 훑어보는 목록이 아니라 한 장씩 사람이 판단할 목록이다.
ISSUE_IMAGE_CAP = 24
FINDINGS_PER_IMAGE = 6
# 사진에 함께 그릴 예측. 다 그리면 의심 지점이 묻힌다.
CONTEXT_CONF = 0.25
CONTEXT_BOXES = 12

LABELS = {
    "missing_label": "라벨 누락 의심",
    "wrong_class": "클래스 오기 의심",
    "phantom_label": "빈 자리 정답",
    "conflicting_gt": "한 물체에 두 클래스",
    "duplicate_gt": "정답 박스 중복",
    "unlabeled_object": "라벨 없는 물체 의심",
}
# 근거가 강한 순. 상한에 걸릴 때 무엇을 먼저 남길지가 이 순서다.
KIND_ORDER = (
    "missing_label",
    "wrong_class",
    "phantom_label",
    "conflicting_gt",
    "duplicate_gt",
    "unlabeled_object",
)
# 모델의 판단을 근거로 쓰는 신호. 모델이 못 믿을 상태면 이것들만 통째로 끈다.
MODEL_KINDS = {"missing_label", "unlabeled_object", "wrong_class", "phantom_label"}

SCOPE_NOTE = (
    "이 후보는 검증(val) 셋에서만 찾은 것입니다. 학습(train) 셋의 라벨은 검사하지 "
    "않았습니다. 또한 여기 오르는 것은 '모델과 라벨이 어긋난 자리' 일 뿐 라벨이 틀렸다는 "
    "증거는 아닙니다 — 사진을 보고 사람이 판단하세요. 라벨을 고쳤다면 그 데이터로 다시 "
    "학습해야 결과에 반영됩니다."
)


def _slices(image_of: np.ndarray, count: int) -> list[tuple[int, int]]:
    """이미지별 행 구간. classify 가 이미지 순서대로 쌓으므로 경계만 찾으면 된다."""
    bounds = [(0, 0)] * count
    start = 0
    for image in range(count):
        end = start
        while end < len(image_of) and image_of[end] == image:
            end += 1
        bounds[image] = (start, end)
        start = end
    return bounds


def _area(box: np.ndarray) -> float:
    """정규화 좌표 기준 넓이(이미지 면적 대비)."""
    return float(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))


def _usable(row: dict[str, Any] | None, key: str, floor: float) -> bool:
    """이 클래스의 모델 판단을 근거로 삼아도 되는가."""
    if row is None or not row.get("evaluated"):
        return False
    value = row.get(key)
    return (
        value is not None
        and value >= floor
        and int(row.get("instances") or 0) >= MIN_CLASS_INSTANCES
    )


def build(
    records: list[dict[str, Any]],
    dets: dict[str, np.ndarray],
    gts: dict[str, np.ndarray],
    names: dict[int, str],
    per_class: list[dict[str, Any]],
    overall: dict[str, Any],
    *,
    conf_reliable: bool,
) -> dict[str, Any]:
    """report.json 의 "label_issues" 에 그대로 들어가는 값. 파일을 읽지 않는다."""
    rows = {int(r["cls"]): r for r in per_class}
    map50 = overall.get("map50")

    # 모델이 이 지경이면 "모델이 틀렸다고 말하는 자리" 를 라벨 오류의 근거로 쓸 수 없다.
    if not conf_reliable:
        reason = (
            "모델이 아직 제대로 학습되지 않아(신뢰도 임계값 추천이 성립하지 않는 상태) "
            "모델 판단을 라벨 근거로 쓸 수 없습니다. 아래에는 라벨만 보고 찾은 것 "
            "(겹치는 정답 박스)만 실었습니다."
        )
    elif map50 is None or map50 < MIN_MODEL_MAP50:
        reason = (
            f"모델의 mAP50 이 {map50 if map50 is not None else 0:.3f} 로 낮아, 모델이 "
            f"'틀렸다' 고 말하는 자리를 라벨 오류의 근거로 쓸 수 없습니다. 아래에는 "
            f"모델과 무관하게 라벨만 보고 찾은 것만 실었습니다."
        )
    else:
        reason = None
    model_evidence = reason is None

    det_at = _slices(dets["img"], len(records))
    gt_at = _slices(gts["img"], len(records))

    findings: list[dict[str, Any]] = []
    for image, record in enumerate(records):
        d0, d1 = det_at[image]
        g0, g1 = gt_at[image]
        gt_cls = gts["cls"][g0:g1]
        gt_taken = gts["taken"][g0:g1]
        gt_display = diagnose.to_display(record["gt_xyxy"], record)

        pred_display: np.ndarray | None = None

        def _pred_box(local: int) -> list[float]:
            nonlocal pred_display
            if pred_display is None:
                matched = diagnose.match(record, 0.0)
                pred_display = diagnose.to_display(matched["p_boxes"], record)
            return [round(float(v), 4) for v in pred_display[local]]

        def _gt_box(local: int) -> list[float]:
            return [round(float(v), 4) for v in gt_display[local]]

        if model_evidence:
            for row in range(d0, d1):
                kind = int(dets["err"][row])
                cls = int(dets["cls"][row])
                conf = float(dets["conf"][row])
                rule = rows.get(cls)
                local = int(dets["idx"][row])

                if kind == tide._CODES["bkg"]:
                    if not _usable(rule, "precision", MIN_CLASS_PRECISION):
                        continue
                    box = _pred_box(local)
                    if _area(np.asarray(box)) < MIN_BOX_AREA:
                        continue
                    same = gt_cls == cls
                    hits = int((same & gt_taken).sum())
                    if same.any() and hits and conf >= HIGH_CONF:
                        findings.append({
                            "image": image, "kind": "missing_label", "cls": cls,
                            "conf": round(conf, 3), "iou": 0.0, "score": conf,
                            "box": box, "ref_box": None, "ref_name": None,
                            "message": (
                                f"이 사진에는 {names.get(cls, cls)} 정답이 {int(same.sum())}개 "
                                f"있고 모델이 그중 {hits}개를 맞췄습니다. 그런데 라벨이 없는 "
                                f"이 자리도 {names.get(cls, cls)} 로 신뢰도 {conf:.2f} 에 "
                                f"검출했습니다. 라벨을 빠뜨린 자리이거나, 학습 클래스에 없는 "
                                f"다른 물체를 {names.get(cls, cls)} 로 잘못 본 것입니다."
                            ),
                        })
                    elif not same.any() and conf >= VERY_HIGH_CONF:
                        findings.append({
                            "image": image, "kind": "unlabeled_object", "cls": cls,
                            "conf": round(conf, 3), "iou": 0.0, "score": conf,
                            "box": box, "ref_box": None, "ref_name": None,
                            "message": (
                                f"이 사진에는 {names.get(cls, cls)} 정답이 하나도 없는데, "
                                f"모델이 이 자리를 {names.get(cls, cls)} 로 신뢰도 "
                                f"{conf:.2f} 에 검출했습니다 (겹치는 정답 박스 없음). 라벨 "
                                f"누락이거나 모델의 오검출입니다 — 사진을 보고 판단하세요."
                            ),
                        })

                elif kind == tide._CODES["cls"]:
                    iou = float(dets["iou"][row])
                    if conf < HIGH_CONF or iou < TIGHT_IOU:
                        continue
                    if not _usable(rule, "precision", MIN_CLASS_PRECISION):
                        continue
                    # 분류상 클래스 오류의 근거 정답은 항상 미매칭이다(tide 의 불변식).
                    truth = int(dets["gt"][row]) - g0
                    findings.append({
                        "image": image, "kind": "wrong_class", "cls": cls,
                        "conf": round(conf, 3), "iou": round(iou, 3), "score": conf * iou,
                        "box": _pred_box(local), "ref_box": _gt_box(truth),
                        "ref_name": names.get(int(gt_cls[truth]), str(gt_cls[truth])),
                        "message": (
                            f"같은 자리(IoU {iou:.2f})의 정답은 "
                            f"{names.get(int(gt_cls[truth]), gt_cls[truth])} 인데 모델은 "
                            f"{names.get(cls, cls)} 로 신뢰도 {conf:.2f} 에 검출했습니다. "
                            f"박스는 맞고 클래스만 다릅니다 — 라벨 클래스를 잘못 지정했을 "
                            f"수 있습니다."
                        ),
                    })

            for local in range(g1 - g0):
                if not gts["miss"][g0 + local]:
                    continue
                if float(gts["best_iou"][g0 + local]) >= NO_SIGNAL_IOU:
                    continue
                cls = int(gt_cls[local])
                if not _usable(rows.get(cls), "recall", MIN_CLASS_RECALL):
                    continue
                box = _gt_box(local)
                if _area(np.asarray(box)) < MIN_BOX_AREA:
                    continue
                recall = rows[cls]["recall"]
                findings.append({
                    "image": image, "kind": "phantom_label", "cls": cls,
                    "conf": None, "iou": 0.0,
                    "score": 1.0 - float(gts["best_iou"][g0 + local]),
                    "box": box, "ref_box": None, "ref_name": None,
                    "message": (
                        f"{names.get(cls, cls)} 정답 박스인데, 신뢰도를 0.001 까지 낮춰도 "
                        f"모델은 이 자리에서 어떤 클래스도 검출하지 못했습니다. "
                        f"{names.get(cls, cls)} 의 재현율은 {recall:.2f} 로 낮지 않습니다 — "
                        f"박스 좌표가 어긋났거나 실제로는 물체가 없는 자리일 수 있습니다."
                    ),
                })

        # 라벨만 보는 신호. 모델이 어떤 상태든 유효하다.
        pairs = diagnose.iou_matrix(record["gt_xyxy"], record["gt_xyxy"])
        for i in range(g1 - g0):
            for j in range(i + 1, g1 - g0):
                iou = float(pairs[i, j])
                if iou < DUP_GT_IOU:
                    continue
                a, b = int(gt_cls[i]), int(gt_cls[j])
                same = a == b
                findings.append({
                    "image": image,
                    "kind": "duplicate_gt" if same else "conflicting_gt",
                    "cls": a, "conf": None, "iou": round(iou, 3), "score": iou,
                    "box": _gt_box(i), "ref_box": _gt_box(j),
                    "ref_name": names.get(b, str(b)),
                    "message": (
                        f"같은 {names.get(a, a)} 정답 박스 두 개가 거의 같은 자리에 "
                        f"있습니다 (IoU {iou:.2f}). 한 물체를 두 번 라벨했을 수 있습니다."
                        if same else
                        f"거의 같은 자리(IoU {iou:.2f})에 {names.get(a, a)} 와 "
                        f"{names.get(b, b)} 정답이 함께 있습니다. 한 물체에 서로 다른 "
                        f"클래스가 붙었을 수 있습니다."
                    ),
                })

    return _assemble(findings, records, names, reason, model_evidence)


def _assemble(
    findings: list[dict[str, Any]],
    records: list[dict[str, Any]],
    names: dict[int, str],
    reason: str | None,
    model_evidence: bool,
) -> dict[str, Any]:
    total = len(findings)
    counts = {kind: 0 for kind in KIND_ORDER}
    for finding in findings:
        counts[finding["kind"]] += 1

    findings.sort(
        key=lambda f: (KIND_ORDER.index(f["kind"]), -f["score"], f["image"])
    )
    taken: dict[str, int] = {kind: 0 for kind in KIND_ORDER}
    kept: list[dict[str, Any]] = []
    for finding in findings:
        if taken[finding["kind"]] >= ISSUE_KIND_CAP:
            continue
        taken[finding["kind"]] += 1
        kept.append(finding)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for finding in kept:
        bucket = grouped.setdefault(finding["image"], [])
        if len(bucket) < FINDINGS_PER_IMAGE:
            bucket.append(finding)

    items = []
    for image in sorted(grouped, key=lambda i: (-max(f["score"] for f in grouped[i]), i))[
        :ISSUE_IMAGE_CAP
    ]:
        record = records[image]
        height, width = record["ori_shape"]
        matched = diagnose.match(record, CONTEXT_CONF)
        order = np.argsort(-matched["p_conf"])[:CONTEXT_BOXES]
        pred_display = diagnose.to_display(matched["p_boxes"][order], record)
        gt_display = diagnose.to_display(record["gt_xyxy"], record)
        items.append({
            "image": record["im_file"],
            "name": Path(record["im_file"]).name,
            "width": int(width),
            "height": int(height),
            "findings": [
                {
                    "kind": f["kind"], "label": LABELS[f["kind"]],
                    "cls": f["cls"], "name": names.get(f["cls"], str(f["cls"])),
                    "conf": f["conf"], "iou": f["iou"], "score": round(f["score"], 4),
                    "box": f["box"], "ref_box": f["ref_box"], "ref_name": f["ref_name"],
                    "message": f["message"],
                }
                for f in grouped[image]
            ],
            "gt": [
                {
                    "cls": int(record["gt_cls"][i]),
                    "name": names.get(int(record["gt_cls"][i]), str(record["gt_cls"][i])),
                    "box": [round(float(v), 4) for v in gt_display[i]],
                    "state": "hit" if matched["gt_taken"][i] >= 0 else "miss",
                }
                for i in range(len(gt_display))
            ],
            "pred": [
                {
                    "cls": int(matched["p_cls"][order[i]]),
                    "name": names.get(
                        int(matched["p_cls"][order[i]]), str(matched["p_cls"][order[i]])
                    ),
                    "conf": round(float(matched["p_conf"][order[i]]), 3),
                    "box": [round(float(v), 4) for v in pred_display[i]],
                    "state": "hit" if matched["p_hit"][order[i]] else "false",
                }
                for i in range(len(pred_display))
            ],
        })

    return {
        "available": True,
        "reason": reason,
        "model_evidence": model_evidence,
        "total": total,
        "shown": sum(len(item["findings"]) for item in items),
        "images_cap": ISSUE_IMAGE_CAP,
        "kinds": [
            {"kind": kind, "label": LABELS[kind], "count": counts[kind]}
            for kind in KIND_ORDER
            if counts[kind]
        ],
        "scope_note": SCOPE_NOTE,
        "items": items,
    }
