"""학습 결과를 "그래서 무엇을 고쳐야 하는가" 로 번역한다.

confusion_matrix.png 을 줘도 비전문가는 다음 행동을 정하지 못한다. 필요한 것은
클래스별로 어디가 약한지, 배포할 때 신뢰도 임계값을 얼마로 둘지, 그리고 실제로 틀린
사진이 어떻게 생겼는지다.

ultralytics 는 검증 중에 이 값을 대부분 이미 계산한다. 다만 매칭이 끝나면 예측 박스를
버리기 때문에(detect/val.py 의 update_metrics), 실패 사례를 보여주려면 그 전에 가로채야 한다.
그래서 검증기를 상속해 원본 예측과 정답을 붙잡아 둔다.

좌표계 주의: 정답은 `xywh2xyxy(bbox) * imgsz` 로 letterbox 픽셀 공간에 놓이고, 예측도 같은
공간에서 나온다(경계 밖으로 삐져나온 값이 있을 수 있으나 공간은 같다). letterbox 는 균일
스케일 + 평행이동이라 IoU 가 원본 좌표계와 같으므로, 계산은 그대로 하고 화면에 그릴 때만
원본 크기로 되돌린다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# 실패 사례 갤러리에 실을 이미지 수 상한. 전부 실으면 report.json 이 수 MB 가 된다.
GALLERY_CAP = 60
# 한 이미지에 그릴 예측 박스 상한. 겹쳐 그리면 아무것도 안 보인다.
BOXES_PER_IMAGE = 40
# 정답과 짝을 지을 때 쓰는 IoU 기준 (COCO 의 mAP50 과 같다).
MATCH_IOU = 0.5
# 신뢰도 곡선을 그대로 실으면 클래스마다 1000점이다. 화면에는 이 정도면 충분하다.
CURVE_POINTS = 101
# 이보다 낮은 임계값을 "최적" 이라고 내놓으면 오검출이 쏟아진다. 추천을 포기하는 선.
MIN_USEFUL_CONF = 0.05
MIN_USEFUL_F1 = 0.10
# 추천을 못 할 때 갤러리에 쓰는 임계값 (추론 화면의 기본값과 같다).
FALLBACK_CONF = 0.25
# 최악 클래스를 몇 개까지 짚어 줄 것인가. 전부 나열하면 무엇부터 손댈지 알 수 없다.
WORST_CAP = 3


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,4) x (m,4) xyxy 박스 쌍의 IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def match(record: dict[str, Any], conf: float) -> dict[str, Any]:
    """이 이미지의 예측을 정답과 짝지어 tp/fp/fn 을 센다.

    신뢰도 높은 예측부터 같은 클래스의 빈 정답 중 IoU 가 가장 큰 것을 가져간다
    (COCO 평가와 같은 탐욕적 방식).
    """
    keep = record["p_conf"] >= conf
    p_boxes, p_cls, p_conf = (
        record["p_xyxy"][keep],
        record["p_cls"][keep],
        record["p_conf"][keep],
    )
    order = np.argsort(-p_conf)
    p_boxes, p_cls, p_conf = p_boxes[order], p_cls[order], p_conf[order]

    g_boxes, g_cls = record["gt_xyxy"], record["gt_cls"]
    ious = iou_matrix(p_boxes, g_boxes)

    gt_taken = np.full(len(g_boxes), -1, dtype=int)  # 정답 -> 이를 맞춘 예측 index
    pred_hit = np.zeros(len(p_boxes), dtype=bool)
    for pi in range(len(p_boxes)):
        best, best_iou = -1, MATCH_IOU
        for gi in range(len(g_boxes)):
            if gt_taken[gi] >= 0 or g_cls[gi] != p_cls[pi]:
                continue
            if ious[pi, gi] >= best_iou:
                best, best_iou = gi, ious[pi, gi]
        if best >= 0:
            gt_taken[best] = pi
            pred_hit[pi] = True

    return {
        "p_boxes": p_boxes,
        "p_cls": p_cls,
        "p_conf": p_conf,
        "p_hit": pred_hit,
        "gt_taken": gt_taken,
        # 오류 분해(tide.py)가 "짝을 못 지은 이유" 를 따지려면 이 행렬이 필요하다.
        # 이미 계산해 놓은 것이라 버리지 않고 넘긴다. 행/열은 위 정렬 순서와 같다.
        "ious": ious,
        "tp": int(pred_hit.sum()),
        "fp": int((~pred_hit).sum()),
        "fn": int((gt_taken < 0).sum()),
    }


def collecting_validator():
    """예측을 버리기 전에 붙잡는 검증기 클래스를 만든다.

    model.val(validator=...) 은 클래스를 받아 스스로 인스턴스화하므로, 결과를 담을 곳을
    클래스 속성으로 들려 보낸다.
    """
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class CollectingValidator(DetectionValidator):
        records: list[dict[str, Any]] = []

        def update_metrics(self, preds, batch):
            for si, pred in enumerate(preds):
                prepared = self._prepare_batch(si, batch)
                CollectingValidator.records.append(
                    {
                        "im_file": str(prepared["im_file"]),
                        "ori_shape": tuple(int(v) for v in prepared["ori_shape"]),
                        "imgsz": tuple(int(v) for v in prepared["imgsz"]),
                        "ratio_pad": prepared["ratio_pad"],
                        "gt_cls": prepared["cls"].cpu().numpy().astype(int).ravel(),
                        "gt_xyxy": prepared["bboxes"].cpu().numpy().astype(np.float32),
                        "p_cls": pred["cls"].cpu().numpy().astype(int).ravel(),
                        "p_conf": pred["conf"].cpu().numpy().astype(np.float32).ravel(),
                        "p_xyxy": pred["bboxes"].cpu().numpy().astype(np.float32),
                    }
                )
            super().update_metrics(preds, batch)

    CollectingValidator.records = []
    return CollectingValidator


def to_display(boxes: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    """letterbox 픽셀 좌표를 원본 기준 0~1 로 되돌린다.

    화면 오버레이는 정규화 좌표를 쓴다(데이터셋 검수 화면과 같은 규약).
    """
    from ultralytics.utils import ops

    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    scaled = ops.scale_boxes(
        record["imgsz"],
        boxes.copy(),
        record["ori_shape"],
        ratio_pad=record["ratio_pad"],
    )
    height, width = record["ori_shape"]
    out = np.asarray(scaled, dtype=np.float32)
    out[:, [0, 2]] /= max(width, 1)
    out[:, [1, 3]] /= max(height, 1)
    return np.clip(out, 0.0, 1.0)


def _finite(value: Any, digits: int = 5) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def confidence_recommendation(box_metric: Any, names: dict[int, str]) -> dict[str, Any]:
    """F1 이 가장 높아지는 신뢰도.

    ultralytics 가 F1_curve.png 에 찍는 그 값과 같은 산식을 쓴다(곡선을 평활한 뒤 최대점).
    추론 기본값 0.25 를 그대로 쓰면 모델마다 손해를 보므로 배포 전에 알아야 한다.
    """
    from ultralytics.utils.metrics import smooth

    px = np.asarray(box_metric.px, dtype=np.float32)
    f1 = np.asarray(box_metric.f1_curve, dtype=np.float32)
    if px.size == 0 or f1.size == 0:
        return {}

    mean_f1 = smooth(f1.mean(0), 0.1)
    best = int(mean_f1.argmax())
    precision = np.asarray(box_metric.p_curve, dtype=np.float32).mean(0)
    recall = np.asarray(box_metric.r_curve, dtype=np.float32).mean(0)

    step = max(1, px.size // CURVE_POINTS)
    per_class = []
    for row, cls_index in enumerate(getattr(box_metric, "ap_class_index", [])):
        if row >= len(f1):
            break
        peak = int(f1[row].argmax())
        per_class.append(
            {
                "cls": int(cls_index),
                "name": names.get(int(cls_index), str(cls_index)),
                "conf": _finite(px[peak]),
                "f1": _finite(f1[row][peak]),
            }
        )

    best_conf = float(px[best])
    best_f1 = float(mean_f1[best])
    # 모델이 나쁘면 F1 은 "전부 검출" 쪽에서 최대가 되어 임계값 0 이 최적이라고 나온다.
    # 산수로는 맞지만 그대로 쓰면 한 장에 수백 개가 잡힌다. 추천 대신 사실을 말한다.
    reliable = best_conf >= MIN_USEFUL_CONF and best_f1 >= MIN_USEFUL_F1
    return {
        "conf": _finite(best_conf),
        "f1": _finite(best_f1),
        "precision": _finite(precision[best]),
        "recall": _finite(recall[best]),
        "reliable": reliable,
        "message": None if reliable else (
            f"F1 이 신뢰도 {best_conf:.3f} 에서 최대({best_f1:.3f})입니다. "
            f"모델이 아직 제대로 학습되지 않아 임계값을 낮출수록 점수가 오르는 상태라, "
            f"이 값을 그대로 쓰면 오검출이 쏟아집니다. 먼저 성능을 올리세요."
        ),
        "per_class": per_class,
        "curve": [
            {"conf": _finite(px[i], 4), "f1": _finite(mean_f1[i], 4)}
            for i in range(0, px.size, step)
        ],
    }


def per_class_table(metrics: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    """클래스별 성능. summary() 를 표준 키로 재포장한다.

    summary() 는 val 에 정답이 하나도 없는 클래스를 통째로 빼 버린다. 그대로 두면
    "표에 없으니 문제 없다" 고 읽히므로, 빠진 클래스를 instances 0 으로 채워 넣는다.
    """
    rows = {}
    for row in metrics.summary(normalize=True, decimals=5):
        rows[str(row.get("Class"))] = row

    table: list[dict[str, Any]] = []
    for cls_index, name in sorted(names.items()):
        row = rows.get(str(name))
        if row is None:
            table.append(
                {
                    "cls": int(cls_index),
                    "name": name,
                    "images": 0,
                    "instances": 0,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "ap50": None,
                    "ap50_95": None,
                    "evaluated": False,
                }
            )
            continue
        table.append(
            {
                "cls": int(cls_index),
                "name": name,
                "images": int(row.get("Images") or 0),
                "instances": int(row.get("Instances") or 0),
                "precision": _finite(row.get("Box-P")),
                "recall": _finite(row.get("Box-R")),
                "f1": _finite(row.get("Box-F1")),
                "ap50": _finite(row.get("mAP50")),
                "ap50_95": _finite(row.get("mAP50-95")),
                "evaluated": True,
            }
        )
    return table


def worst_classes(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """어느 클래스부터 손대야 하는지.

    모든 클래스가 약하면 그건 클래스 문제가 아니라 모델 문제다. 그때 전부 나열하면
    무엇부터 해야 할지 오히려 알 수 없어지므로, 그 사실을 한 줄로 말하고 만다.
    """
    # 검증 셋에 정답이 없어 평가조차 못 한 클래스는 성능을 알 수 없다 — 먼저 알린다.
    unevaluated = [
        {
            "name": row["name"],
            "ap50_95": None,
            "instances": 0,
            "message": f"{row['name']} 은 검증 셋에 정답이 하나도 없어 평가되지 않았습니다. "
            f"성능을 알 수 없으니 검증 셋에 포함시키세요.",
        }
        for row in table
        if not row["evaluated"]
    ]

    scored = [r for r in table if r["evaluated"] and r["ap50_95"] is not None]
    if not scored:
        return unevaluated

    median = float(np.median([r["ap50_95"] for r in scored]))
    weak = [r for r in scored if r["ap50_95"] < 0.3 or r["ap50_95"] < median * 0.5]

    if len(weak) == len(scored) and len(scored) > 1:
        return unevaluated + [
            {
                "name": None,
                "ap50_95": round(median, 5),
                "instances": 0,
                "message": f"특정 클래스가 아니라 모든 클래스가 약합니다 "
                f"(mAP50-95 중앙값 {median:.3f}). 클래스별로 손볼 단계가 아니라 "
                f"학습을 더 하거나 데이터 전체를 늘려야 합니다.",
            }
        ]

    worst = []
    for row in sorted(weak, key=lambda r: (r["ap50_95"], r["instances"]))[:WORST_CAP]:
        if row["instances"] < 100:
            reason = f"인스턴스가 {row['instances']}개로 적습니다. 데이터를 늘리는 편이 빠릅니다."
        else:
            reason = "데이터는 충분한데 성능이 낮습니다. 라벨 기준이 흔들리지 않는지 확인하세요."
        worst.append(
            {
                "name": row["name"],
                "ap50_95": row["ap50_95"],
                "instances": row["instances"],
                "message": f"{row['name']} 의 mAP50-95 가 {row['ap50_95']:.3f} 입니다 "
                f"(전체 중앙값 {median:.3f}). {reason}",
            }
        )
    return unevaluated + worst


def build_gallery(
    records: list[dict[str, Any]], conf: float, names: dict[int, str]
) -> tuple[list[dict[str, Any]], int]:
    """가장 많이 틀린 이미지부터.

    보여주는 예측은 추천 신뢰도 이상만 남긴다. 검증은 conf 0.001 로 돌기 때문에 한 장에
    수백 개가 잡히는데, 그걸 다 그리면 사진이 박스로 뒤덮여 아무것도 판단할 수 없다.
    실제로 배포할 임계값에서 무엇이 틀리는지가 사용자가 알고 싶은 것이다.
    """
    scored = []
    for record in records:
        matched = match(record, conf)
        # 놓친 것을 오검출보다 무겁게 본다 — 사용자가 더 아프게 느낀다.
        score = matched["fn"] * 1.5 + matched["fp"]
        if score > 0:
            scored.append((score, record, matched))
    scored.sort(key=lambda item: (-item[0], item[1]["im_file"]))

    gallery = []
    for score, record, matched in scored[:GALLERY_CAP]:
        height, width = record["ori_shape"]
        order = np.argsort(-matched["p_conf"])[:BOXES_PER_IMAGE]
        p_display = to_display(matched["p_boxes"][order], record)
        g_display = to_display(record["gt_xyxy"], record)

        gallery.append(
            {
                "image": record["im_file"],
                "name": Path(record["im_file"]).name,
                "width": int(width),
                "height": int(height),
                "score": round(float(score), 2),
                "tp": matched["tp"],
                "fp": matched["fp"],
                "fn": matched["fn"],
                "gt": [
                    {
                        "cls": int(record["gt_cls"][i]),
                        "name": names.get(
                            int(record["gt_cls"][i]), str(record["gt_cls"][i])
                        ),
                        "box": [round(float(v), 4) for v in g_display[i]],
                        "state": "hit" if matched["gt_taken"][i] >= 0 else "miss",
                    }
                    for i in range(len(g_display))
                ],
                "pred": [
                    {
                        "cls": int(matched["p_cls"][order[i]]),
                        "name": names.get(
                            int(matched["p_cls"][order[i]]),
                            str(matched["p_cls"][order[i]]),
                        ),
                        "conf": _finite(matched["p_conf"][order[i]], 3),
                        "box": [round(float(v), 4) for v in p_display[i]],
                        "state": "hit" if matched["p_hit"][order[i]] else "false",
                    }
                    for i in range(len(p_display))
                ],
            }
        )
    return gallery, len(scored)
