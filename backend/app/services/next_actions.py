"""진단 결과를 "그래서 다음에 무엇을 할 것인가" 한두 문장으로 바꾼다.

표와 그래프를 아무리 잘 만들어도 비전문가는 거기서 행동을 끌어내지 못한다. 오류 분해까지
와서 "놓침이 지배적" 이라는 사실을 알려 줘도, 그래서 에폭을 늘려야 하는지 데이터를 늘려야
하는지는 별개의 지식이다. 그 마지막 한 걸음을 여기서 잇는다.

**리포트 파일에 굳히지 않고 요청받을 때마다 계산한다.** 근거(report.json)가 run 폴더에
영구히 남아 언제든 재계산할 수 있고, 규칙을 고치면 과거 리포트에도 즉시 적용되기 때문이다.
굳혀 두면 문구 한 줄 고치자고 수 분짜리 검증을 다시 돌려야 한다 — diagnose_fail 과 같은 이유다.

report.json 딕셔너리 하나만 본다. 파일도 DB 도 읽지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.services import quality, tide

# 넷 이상 늘어놓으면 우선순위가 사라져 아무것도 안 하게 된다.
ACTION_CAP = 3
# 한 유형이 전체 손실의 이만큼을 차지하면 "지배적" 이라고 부른다.
DOMINANT_SHARE = 0.35
DOMINANT_SHARE_SOFT = 0.30
# "고쳐서 얻을 게 있는가" 의 기준은 오류 분해와 같은 값을 써야 한다. 두 화면이 서로 다른
# 선을 쓰면 카드 하나는 "에폭을 늘리세요", 다른 하나는 "고칠 것 없습니다" 가 된다.
MIN_ACTIONABLE_DAP = tide.MIN_ACTIONABLE_DAP
# 이보다 나쁘면 어떤 오류가 많은지 따지는 게 의미가 없다.
MIN_USABLE_MAP50 = 0.10
# 추론 화면 기본값. 여기서 이만큼 떨어져 있으면 알려 준다.
DEFAULT_CONF = 0.25
CONF_GAP = 0.10


@dataclass(frozen=True)
class Action:
    code: str
    severity: str  # critical | warn | info
    when: Callable[[dict[str, Any]], bool]
    title: str
    cause: str
    fix: str
    # 참이면 이것 하나만 내고 끝낸다. 다른 처방이 의미를 잃는 상황.
    terminal: bool = False


class _Safe(dict):
    """자리표시자가 비어도 문장이 깨지지 않게 한다(예전 리포트에는 없는 값이 있다)."""

    def __missing__(self, key: str) -> str:
        return "-"


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _context(report: dict[str, Any], data_quality: dict[str, Any] | None = None) -> dict[str, Any]:
    overall = report.get("overall") or {}
    recommendation = report.get("conf_recommendation") or {}
    tide = report.get("tide") or {}
    labels = report.get("label_issues") or {}

    errors = tide.get("errors") or [] if not tide.get("failed") else []
    dap = {e["kind"]: (e.get("dap") or 0.0) for e in errors}
    count = {e["kind"]: int(e.get("count") or 0) for e in errors}
    # 배포 임계값에서 실제로 보이는 건수. 없는(예전) 리포트면 전체 건수로 물러선다.
    seen = {
        e["kind"]: int(e["count_at_conf"] if e.get("count_at_conf") is not None else (e.get("count") or 0))
        for e in errors
    }
    total = sum(v for v in dap.values() if v > 0)
    dominant = max(dap, key=lambda k: dap[k]) if total > 0 else None

    weak = [
        w["name"] for w in (report.get("worst_classes") or [])
        if w.get("name") and (w.get("ap50_95") is not None)
    ]
    pairs = tide.get("confusion_pairs") or []
    instances = int(overall.get("instances") or 0)
    label_total = int(labels.get("total") or 0)

    # 데이터 품질 검사는 별도 잡이라 안 돌렸을 수 있다. 그때는 누수 규칙이 조용히 꺼진다 —
    # 확인하지 않은 것을 없다고 말하지 않는다.
    leak = (data_quality or {}).get("leakage") or {}
    leak_ratio = float(leak.get("ratio") or 0.0) if not leak.get("failed") else 0.0
    leak_count = int(leak.get("val_leaked") or 0) if not leak.get("failed") else 0

    context: dict[str, Any] = {
        "leak_ratio": leak_ratio,
        "leak_count": leak_count,
        "leak_pct": f"{leak_ratio * 100:.1f}",
        "leak_val_total": int(leak.get("val_total") or 0),
        "map50": overall.get("map50"),
        "precision": overall.get("precision"),
        "recall": overall.get("recall"),
        "instances": instances,
        "conf_reliable": bool(recommendation.get("reliable")),
        "conf": recommendation.get("conf"),
        "dap": dap,
        "count": count,
        "seen": seen,
        "dominant": dominant,
        "dominant_share": (dap[dominant] / total) if dominant else 0.0,
        "dominant_value": dap[dominant] if dominant else 0.0,
        "weak_named": weak,
        "label_total": label_total,
        "label_usable": bool(labels.get("available")) and bool(labels.get("model_evidence")),
        "confusion_count": pairs[0]["count"] if pairs else 0,
        # 문장에 그대로 박히는 표시용 문자열.
        "map50_s": _fmt(overall.get("map50")),
        "map50_95_s": _fmt(overall.get("map50_95")),
        "precision_s": _fmt(overall.get("precision")),
        "recall_s": _fmt(overall.get("recall")),
        "conf_s": _fmt(recommendation.get("conf"), 2),
        "dominant_dap": _fmt(dap.get(dominant) if dominant else None),
        "dominant_pct": f"{(dap[dominant] / total) * 100:.0f}" if dominant else "-",
        "dupe_count": seen.get("dupe", 0),
        "label_count": label_total,
        "label_pct": f"{(label_total / instances) * 100:.0f}" if instances else "-",
        "weak_names": ", ".join(weak) if weak else "-",
        "confusion_top": f"{pairs[0]['pred']} ↔ {pairs[0]['gt']}" if pairs else "-",
    }
    return context


def _is(kind: str, floor: float = DOMINANT_SHARE_SOFT) -> Callable[[dict[str, Any]], bool]:
    """이 유형이 지배적이고, 고쳐서 얻을 게 실제로 있는가."""
    return lambda c: (
        c["dominant"] == kind
        and c["dominant_share"] >= floor
        and c["dominant_value"] >= MIN_ACTIONABLE_DAP
    )


ACTIONS: list[Action] = [
    Action(
        # 맨 앞이다. 아래 처방들은 전부 mAP 를 근거로 삼는데 그 mAP 자체가 부풀려진
        # 상황이기 때문이다. 다만 terminal 은 아니다 — 어떤 오류 "유형" 이 많은가는
        # 누수와 무관하게 여전히 참이다.
        "val_leakage",
        "critical",
        lambda c: c["leak_ratio"] >= quality.LEAK_RATIO_ALERT,
        "지금 점수를 그대로 믿으면 안 됩니다",
        "검증용 사진 {leak_val_total}장 가운데 {leak_count}장({leak_pct}%)이 학습용에도 "
        "들어 있습니다. 모델이 이미 외운 사진으로 채점하고 있어서 mAP50 {map50_s} 는 "
        "처음 보는 사진에서의 실력보다 높게 나온 값입니다.",
        "데이터셋 화면의 품질 검사에서 겹치는 사진을 확인하고, 검증용 쪽을 지운 뒤 "
        "다시 학습하세요. 그 전에는 이 아래 숫자들의 절대값을 신뢰하지 마세요 "
        "(어떤 오류가 많은가의 비교는 그대로 쓸 수 있습니다).",
    ),
    Action(
        "model_not_ready",
        "critical",
        lambda c: c["map50"] is None or c["map50"] < MIN_USABLE_MAP50,
        "아직 오류 유형을 따질 단계가 아닙니다",
        "전체 mAP50 이 {map50_s} 입니다. 이 상태에서 어떤 오류가 많은지 세어 봐야 "
        "'모델이 아직 아무것도 제대로 배우지 못했다' 는 한 가지 결론밖에 나오지 않습니다.",
        "새 학습을 만들어 에폭을 늘리고(최소 100), 조기 종료(patience)를 켜서 끝까지 "
        "돌리세요. 그래도 오르지 않으면 학습 이미지 수와 라벨 품질부터 확인해야 합니다.",
        terminal=True,
    ),
    Action(
        # mAP 자체는 볼만한데 쓸 만한 임계값이 없는 상태. 원인이 다르니 문장도 달라야 한다.
        "threshold_unusable",
        "critical",
        lambda c: not c["conf_reliable"],
        "점수는 나오지만 아직 배포할 수 없습니다",
        "mAP50 은 {map50_s} 인데, F1 이 신뢰도를 낮출수록 계속 오릅니다. 쓸 만한 임계값이 "
        "없다는 뜻이라 지금 배포하면 한 장에 수십~수백 개가 잡힙니다. 이 상태에서는 오류 "
        "유형을 나눠 봐야 '아직 덜 배웠다' 는 결론으로 수렴합니다.",
        "에폭을 늘려 더 학습시키세요. 그러고도 임계값이 잡히지 않으면 클래스별 성능표에서 "
        "특정 클래스가 전체를 끌어내리고 있지 않은지 확인하세요.",
        terminal=True,
    ),
    Action(
        "labels_suspect",
        "warn",
        lambda c: c["label_usable"]
        and c["label_total"] >= max(10, c["instances"] * 0.02),
        "파라미터보다 라벨을 먼저 보세요",
        "검증 셋 정답 {instances}개 중 {label_count}건({label_pct}%)이 모델 판단과 "
        "어긋납니다. 이 정도면 학습 설정을 바꿔도 점수가 잘 오르지 않습니다.",
        "아래 '라벨 오류 후보' 카드의 사진을 확인하세요. 라벨을 고쳤다면 데이터셋을 다시 "
        "등록하고 학습을 다시 돌려야 반영됩니다. 다만 이 후보는 검증 셋에서만 찾은 것이라 "
        "학습 셋에도 같은 문제가 있을 가능성이 큽니다.",
    ),
    Action(
        "miss_dominant",
        "warn",
        _is("miss", DOMINANT_SHARE),
        "있는 것을 놓치는 게 가장 큽니다",
        "놓친 정답이 mAP50 을 {dominant_dap} 깎아 전체 손실의 {dominant_pct}% 를 "
        "차지합니다. 재현율이 {recall_s} 입니다 — 모델이 물체 자체를 찾지 못하고 있습니다.",
        "순서대로 시도하세요. ① 배포 신뢰도를 {conf_s} 보다 낮게 잡으면 즉시 더 잡습니다"
        "(대신 오검출이 늡니다). ② 작은 물체가 많다면 imgsz 를 한 단계 올리세요(640→960). "
        "③ 그래도 안 되면 놓친 사진과 비슷한 데이터를 늘리는 것이 근본 대책입니다.",
    ),
    Action(
        "bkg_dominant",
        "warn",
        _is("bkg", DOMINANT_SHARE),
        "없는 것을 있다고 합니다",
        "배경 오검출이 mAP50 을 {dominant_dap} 깎아 전체 손실의 {dominant_pct}% 를 "
        "차지합니다. 정밀도가 {precision_s} 입니다.",
        "① 배포 신뢰도를 {conf_s} 이상으로 올리면 대부분 사라집니다(대신 놓치는 게 늡니다). "
        "② 오검출이 특정 배경에서 반복되면 그 장면의 라벨 없는 이미지를 학습 데이터에 "
        "넣으세요 — 배경을 배경으로 배웁니다. ③ '라벨 오류 후보' 에 같은 자리가 올라와 "
        "있다면 오검출이 아니라 라벨 누락입니다.",
    ),
    Action(
        "loc_dominant",
        "warn",
        _is("loc"),
        "찾긴 하는데 박스가 어긋납니다",
        "위치 오류가 mAP50 을 {dominant_dap} 깎습니다. 물체는 제대로 알아보는데 경계가 "
        "맞지 않는 상태입니다 — mAP50 은 그럭저럭인데 mAP50-95({map50_95_s})가 낮게 "
        "나오는 전형적인 모습입니다.",
        "① imgsz 를 올리면 경계가 또렷해집니다(640→960). ② 회전·스케일 증강(degrees, "
        "scale)이 강하면 낮추고, 마지막 몇 에폭은 close_mosaic 으로 모자이크를 끄세요. "
        "③ 에폭을 늘리면 박스 회귀는 늦게까지 개선됩니다. ④ 라벨 박스 자체가 헐겁게 "
        "그려져 있지 않은지도 확인하세요.",
    ),
    Action(
        "cls_dominant",
        "warn",
        lambda c: _is("cls")(c) and c["confusion_count"] > 0,
        "물체는 찾는데 종류를 혼동합니다",
        "클래스 오류가 mAP50 을 {dominant_dap} 깎습니다. 가장 많이 헷갈리는 조합은 "
        "{confusion_top}({confusion_count}건)입니다.",
        "① 두 클래스의 라벨 기준이 사람 사이에서도 갈리는지 보세요 — 갈린다면 모델도 "
        "못 나눕니다. ② 실제로 구분할 필요가 없다면 한 클래스로 합치는 편이 정확도·운영 "
        "양쪽에 낫습니다. ③ 구분해야 한다면 그 두 클래스의 경계 사례 데이터를 늘리세요. "
        "에폭이나 해상도로는 잘 해결되지 않습니다.",
    ),
    Action(
        "cls_dominant_plain",
        "warn",
        lambda c: _is("cls")(c) and c["confusion_count"] == 0,
        "물체는 찾는데 종류를 혼동합니다",
        "클래스 오류가 mAP50 을 {dominant_dap} 깎습니다. 박스는 물체에 맞는데 종류를 "
        "잘못 부르고 있습니다.",
        "헷갈리는 두 클래스의 라벨 기준이 사람 사이에서도 갈리는지 보세요. 실제로 구분할 "
        "필요가 없다면 한 클래스로 합치는 편이 낫고, 구분해야 한다면 경계 사례 데이터를 "
        "늘리세요. 에폭이나 해상도로는 잘 해결되지 않습니다.",
    ),
    Action(
        "both_dominant",
        "warn",
        _is("both"),
        "위치도 종류도 어긋납니다",
        "위치·클래스가 함께 틀린 오류가 mAP50 을 {dominant_dap} 깎습니다. 이건 사실상 "
        "'엉뚱한 것을 엉뚱한 자리에서 봤다' 는 뜻으로, 특정 축을 손봐서 나아지는 종류가 "
        "아닙니다.",
        "학습이 충분히 되지 않았을 가능성이 큽니다. 에폭을 늘리고(patience 를 켠 채로) "
        "다시 돌리세요. 그래도 남으면 데이터가 부족한 것입니다.",
    ),
    Action(
        "dupe_notable",
        "info",
        # 전체 검출 기준으로 세면 conf 0.001 짜리 잡음이 수백 건 잡혀 멀쩡한 모델에도 뜬다.
        lambda c: c["dap"].get("dupe", 0) >= MIN_ACTIONABLE_DAP
        or c["seen"].get("dupe", 0) >= max(10, c["instances"] * 0.05),
        "같은 물체를 두 번 잡습니다",
        "중복 검출이 {dupe_count}건 있습니다. 한 물체에 박스가 여러 개 붙으면 개수를 세는 "
        "용도에서는 그대로 틀린 답이 됩니다.",
        "추론할 때 NMS IoU 를 낮추면 줄어듭니다. 함께 '라벨 오류 후보' 의 '정답 박스 중복' "
        "항목을 보세요 — 학습 라벨 자체가 겹쳐 있으면 모델이 겹치게 내도록 배웁니다.",
    ),
    Action(
        "conf_far_from_default",
        "info",
        lambda c: c["conf_reliable"]
        and c["conf"] is not None
        and abs(c["conf"] - DEFAULT_CONF) >= CONF_GAP,
        "배포 신뢰도 기본값을 그대로 쓰면 손해입니다",
        "F1 이 가장 높아지는 신뢰도는 {conf_s} 인데 추론 화면 기본값은 0.25 입니다.",
        "추론·내보내기에서 신뢰도를 {conf_s} 로 잡으세요. 놓치는 것이 더 아픈 용도라면 "
        "이보다 낮게, 오검출이 더 아프면 높게 조정하세요.",
    ),
    Action(
        "one_weak_class",
        "info",
        lambda c: bool(c["weak_named"]) and c["dominant_share"] < DOMINANT_SHARE,
        "특정 클래스만 뒤처집니다",
        "{weak_names} 의 성능이 다른 클래스보다 뚜렷하게 낮습니다.",
        "위 '클래스별 성능' 표의 안내를 따르세요. 인스턴스가 적으면 데이터를 늘리는 것이, "
        "충분한데도 낮으면 라벨 기준을 통일하는 것이 먼저입니다.",
    ),
]

# 손볼 곳을 짚어 주는 처방(warn 이상)이 하나도 없을 때만 낸다. "고칠 게 없다" 는 판단이라
# 다른 처방과 나란히 두면 서로를 부정한다.
FALLBACK = Action(
    "looks_healthy",
    "info",
    lambda c: True,
    "지금 크게 고칠 것은 없습니다",
    "mAP50 {map50_s}, 오류가 특정 유형에 몰려 있지 않고 어느 하나를 고쳐도 상승분이 "
    "미미합니다. 한 가지를 손봐서 크게 오르는 상태가 아닙니다.",
    # 예전에는 "데이터를 늘리는 것이 가장 확실하고, 그다음이 더 큰 모델(yolo11s 등)" 이라고
    # 순위를 단정했다. 실측이 뒷순위를 부정한다 — brain-tumor 를 시드 3개씩 돌려 보니
    # yolo11n 대비 s 가 +0.0008, m 이 +0.0029 로 둘 다 문턱(0.024) 근처에도 못 갔는데
    # 시간은 1.5배·2.9배였다 (.codex/phase-6.md 블록 G). 그래서 순위를 빼고 비용을 밝힌다.
    "지금 성능으로 충분한지는 용도가 정합니다. 더 올리려면 데이터를 늘리는 것이 가장 "
    "확실합니다. 더 큰 모델(yolo11s·yolo11m)은 brain-tumor 한 데이터셋을 30에폭으로 "
    "재본 결과로는 mAP 를 흔들림 이상으로 올리지 못하면서 학습 시간만 1.5~2.9배가 "
    "됐습니다. 표본이 하나뿐이니 단정할 수는 없지만 기대는 낮게 잡으세요.",
)


def build(
    report: dict[str, Any], data_quality: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """report.json 딕셔너리만 보고 처방을 만든다. 파일 I/O 없음.

    data_quality 는 데이터셋 품질 검사(quality.json)의 결과다. 그 잡을 안 돌렸으면
    None 이고, 누수 규칙만 조용히 꺼진다. 읽어 오는 것은 호출자(API)의 몫이다 —
    이 함수의 순수성이 규칙을 시험 가능하게 만든다.
    """
    context = _context(report, data_quality)

    def render(action: Action) -> dict[str, Any]:
        return {
            "code": action.code,
            "severity": action.severity,
            "title": action.title.format_map(_Safe(context)),
            "cause": action.cause.format_map(_Safe(context)),
            "fix": action.fix.format_map(_Safe(context)),
        }

    out: list[dict[str, Any]] = []
    for action in ACTIONS:
        try:
            hit = action.when(context)
        except (TypeError, KeyError):
            # 예전 리포트에는 없는 값이 있다. 그 규칙만 건너뛴다.
            continue
        if not hit:
            continue
        if action.terminal:
            return [render(action)]
        out.append(render(action))
        if len(out) >= ACTION_CAP:
            break

    if not any(a["severity"] in ("critical", "warn") for a in out):
        out.append(render(FALLBACK))
    return out[:ACTION_CAP]
