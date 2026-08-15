"""학습 결과 진단 워커 — 검증을 한 번 더 돌려 "무엇이 왜 틀리는가" 를 만든다.

학습 워커·내보내기 워커와 같은 규약을 따른다: 독립 프로세스, jsonl 로 진행 상황 append,
stop.flag 로 취소.

검증 한 번으로 클래스별 성능·최적 신뢰도·실패 사례를 모두 만든다. 나눠서 돌리면 같은
데이터를 두 번 추론하게 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import apply_offline_env  # noqa: E402

apply_offline_env()


def write(path: Path, payload: dict) -> None:
    payload.setdefault("ts", time.time())
    # 한 줄을 한 번의 write 로 내보내야 리더가 잘린 JSON 을 보지 않는다.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events = Path(args.events).resolve()
    weights = (run_dir / args.weights).resolve()

    write(events, {
        "t": "start", "weights": args.weights, "imgsz": args.imgsz,
        "device": args.device, "batch": args.batch,
    })

    started = time.time()
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        data_yaml = config["data"]

        write(events, {"t": "progress", "stage": "val", "message": "검증 예측을 모으는 중"})

        from ultralytics import YOLO

        from app.services import diagnose, label_issues, tide

        validator = diagnose.collecting_validator()
        model = YOLO(str(weights))
        # conf 를 낮게 잡아야 신뢰도 곡선 전체가 그려진다. 화면에 보여줄 때만 다시 거른다.
        metrics = model.val(
            data=data_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=0.001,
            plots=False,
            verbose=False,
            validator=validator,
            project=str(out_dir),
            name="val",
            exist_ok=True,
        )
        records = validator.records

        write(events, {
            "t": "progress", "stage": "analyze",
            "message": f"{len(records)}장의 예측을 분석하는 중",
        })

        names = {int(k): str(v) for k, v in (metrics.names or {}).items()}
        table = diagnose.per_class_table(metrics, names)
        recommendation = diagnose.confidence_recommendation(metrics.box, names)
        # 갤러리는 "실제로 배포할 임계값에서 무엇이 틀리는가" 를 보여준다.
        # 추천값을 믿을 수 없을 때(모델이 덜 학습돼 임계값 0 이 최적으로 나오는 경우)
        # 그대로 쓰면 한 장에 수백 개가 그려지므로 추론 화면 기본값으로 돌아간다.
        gallery_conf = (
            recommendation["conf"]
            if recommendation.get("reliable") and recommendation.get("conf") is not None
            else diagnose.FALLBACK_CONF
        )
        gallery, gallery_total = diagnose.build_gallery(records, gallery_conf, names)

        write(events, {
            "t": "progress", "stage": "tide", "message": "오류 유형을 나누는 중",
        })
        # 분해가 깨져도 나머지 리포트는 살아야 한다. 다만 조용히 빼면 "예전 리포트라 없는 것"
        # 과 구분되지 않으므로, 실패했다는 사실 자체를 값으로 남긴다.
        results = getattr(metrics, "results_dict", {}) or {}
        overall = {
            "images": len(records),
            "instances": int(sum(len(r["gt_cls"]) for r in records)),
            "precision": _num(results.get("metrics/precision(B)")),
            "recall": _num(results.get("metrics/recall(B)")),
            "map50": _num(results.get("metrics/mAP50(B)")),
            "map50_95": _num(results.get("metrics/mAP50-95(B)")),
        }
        try:
            # 분류는 한 번만 한다. 두 화면이 서로 다른 판정을 보여 주면 안 된다.
            classified = tide.classify(records)
            breakdown = tide.error_breakdown(
                records, names, collection_conf=0.001, classified=classified,
                # 건수는 실제로 배포할 임계값에서 세야 의미가 있다. 갤러리와 같은 값을 쓴다.
                deploy_conf=gallery_conf,
            )
            issues = label_issues.build(
                records, *classified, names, table, overall,
                conf_reliable=bool(recommendation.get("reliable")),
            )
        except Exception as exc:  # noqa: BLE001
            breakdown = {
                "failed": True,
                "message": f"오류 분해를 계산하지 못했습니다: {exc}",
            }
            issues = {
                "available": False,
                "reason": f"라벨 오류 후보를 찾지 못했습니다: {exc}",
                "model_evidence": False,
                "total": 0, "shown": 0, "kinds": [], "items": [],
                "scope_note": label_issues.SCOPE_NOTE,
                "images_cap": label_issues.ISSUE_IMAGE_CAP,
            }

        report = {
            "schema_version": 3,
            "run_id": run_dir.name,
            "dataset_id": config.get("dataset_id"),
            "weights": args.weights,
            "device": args.device,
            "created_at": time.time(),
            "elapsed_s": round(time.time() - started, 1),
            "params": {"imgsz": args.imgsz, "conf": 0.001, "batch": args.batch},
            "classes": [names[k] for k in sorted(names)],
            "overall": overall,
            "tide": breakdown,
            "label_issues": issues,
            "per_class": table,
            "worst_classes": diagnose.worst_classes(table),
            "conf_recommendation": recommendation,
            "gallery": gallery,
            "gallery_total": gallery_total,
            "gallery_cap": diagnose.GALLERY_CAP,
            "gallery_conf": gallery_conf,
        }
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
        write(events, {
            "t": "end", "status": "completed", "report": "report.json",
            "elapsed_s": round(time.time() - started, 1),
        })
        return 0
    except Exception as exc:  # noqa: BLE001 - 실패도 사용자에게 보여야 한다
        write(events, {
            "t": "end", "status": "failed", "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return 1


def _num(value) -> float | None:
    import math

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 5) if math.isfinite(number) else None


if __name__ == "__main__":
    raise SystemExit(main())
