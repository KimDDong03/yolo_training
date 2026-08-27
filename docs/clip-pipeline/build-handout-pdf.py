# -*- coding: utf-8 -*-
"""데이터과 전달용 기획서 PDF — 1부(재작성 원고) + 부록 A(규격서 변환)."""

import re
from pathlib import Path
import markdown  # type: ignore[import-untyped]  # tensorboard 동반 설치분 — 문서 빌드 전용

DOCS = Path(__file__).resolve().parent
OUT = DOCS / "_handout.html"
# PDF 생성:
#   python build-handout-pdf.py
#   msedge --headless=new --disable-gpu --no-pdf-header-footer \
#     --print-to-pdf="<이 폴더>\CCTV_클립_학습데이터_파이프라인_기획서_v1.pdf" "file:///<이 폴더>/_handout.html"

# ---------- 부록 A 변환 ----------
spec = (DOCS / "interface-spec.md").read_text(encoding="utf-8")
spec = re.sub(r"\((?:\./)?decisions\.md(#[-\w]+)\)", r"(\1)", spec)
spec = re.sub(r"\[([^\]]+)\]\((?:\./)?(?:README|decisions)\.md\)", r"\1", spec)
spec = re.sub(r"\[([^\]]+)\]\(#\d[^)]*\)", r"\1", spec)
spec_html = markdown.Markdown(extensions=["tables", "fenced_code"]).convert(spec)

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
* { box-sizing: border-box; }
body { font-family: 'Pretendard','Malgun Gothic',sans-serif; font-size: 10pt; line-height: 1.66;
       color: #111; margin: 0; word-break: keep-all; }
h1,h2,h3,h4 { color: #111; line-height: 1.35; }
p { margin: 6pt 0; }
ul, ol { margin: 5pt 0 9pt; padding-left: 19pt; }
li { margin: 2.5pt 0; }
a { color: #111; text-decoration: none; }
strong, b { font-weight: 700; }
code { font-family: Consolas,'Pretendard',monospace; font-size: 8.8pt; background: #f2f2f2;
       padding: 0.5pt 3.5pt; border-radius: 2px; overflow-wrap: anywhere; }
pre { font-family: Consolas,'Pretendard',monospace; font-size: 8.2pt; line-height: 1.45;
      background: #f7f7f7; border: 1px solid #bbb; padding: 8pt 10pt;
      white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { margin: 8pt 0; padding: 5pt 12pt; border: 1px solid #bbb; background: #fafafa; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9pt; }
th, td { border: 1px solid #888; padding: 4.5pt 7pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; color: #111; white-space: nowrap; font-weight: 700; }
tr { page-break-inside: avoid; }
thead { display: table-header-group; }
td:first-child code { white-space: nowrap; overflow-wrap: normal; }
h2, h3, h4 { page-break-after: avoid; }

/* ---------- 표지 ---------- */
.cover { page-break-after: always; padding-top: 84pt; }
.kicker { font-size: 10.5pt; color: #111; letter-spacing: 3px; font-weight: 700; margin-bottom: 22pt; }
.cover h1 { font-size: 25pt; margin: 0 0 12pt; }
.cover .sub { font-size: 11pt; color: #444; margin-bottom: 40pt; }
.cover-meta { font-size: 10pt; }
.cover-meta th { width: 84pt; }
.howto { margin-top: 26pt; }

/* ---------- 1부 공통 ---------- */
.sec { page-break-before: always; }
.sec h2 { font-size: 15.5pt; margin: 0 0 12pt; padding-bottom: 5pt; border-bottom: 2px solid #111; }
.sec h2 .no { color: #888; margin-right: 9pt; }
h3 { font-size: 11.8pt; margin: 15pt 0 6pt; }

.callout { border: 1px solid #999; padding: 8pt 12pt; margin: 10pt 0; page-break-inside: avoid; }
.callout.warn, .callout.done { border: 1px solid #999; background: none; }
.callout .ct { font-weight: 700; }

.stats { display: flex; gap: 8pt; margin: 12pt 0; }
.stat { flex: 1; border: 1px solid #999; padding: 8pt 10pt; text-align: center; page-break-inside: avoid; }
.stat .v { font-size: 15pt; font-weight: 800; color: #111; }
.stat .k { font-size: 8.5pt; color: #555; margin-top: 2pt; line-height: 1.4; }

/* ---------- 흐름도 ---------- */
.flow { margin: 12pt 0; }
.zone { border: 1px solid #777; padding: 9pt 12pt; margin: 0; page-break-inside: avoid; background: #fff; }
.zone.hl { border: 1.6px solid #111; }
.zone .zt { font-weight: 800; color: #111; font-size: 10.5pt; }
.zone .zs { font-size: 8.8pt; color: #555; }
.steps { display: flex; flex-wrap: wrap; gap: 5pt; margin-top: 7pt; }
.step { background: #f2f2f2; border: 1px solid #aaa; padding: 3pt 8pt;
        font-size: 8.8pt; color: #222; font-weight: 500; }
.flow-arrow { text-align: center; color: #333; font-size: 9pt; padding: 3pt 0; }
.flow-arrow b { color: #111; }

/* ---------- 단계 블록 ---------- */
.stage { margin: 0 0 13pt; page-break-inside: avoid; }
.stage .sh { font-weight: 800; font-size: 11pt; color: #111; margin-bottom: 3pt; }
.stage .sh .n { display: inline-block; min-width: 16pt; height: 16pt; line-height: 16pt; text-align: center;
                background: #111; color: #fff; border-radius: 50%; font-size: 9pt; margin-right: 6pt; }

/* ---------- D 카드 ---------- */
.dcard { border: 1px solid #888; margin: 0 0 11pt; page-break-inside: avoid; }
.dcard .dh { display: flex; align-items: center; gap: 8pt; background: #f0f0f0; padding: 5.5pt 11pt;
             border-bottom: 1px solid #888; }
.dcard .did { font-weight: 800; color: #111; font-size: 10.5pt; }
.dcard .dt { font-weight: 700; font-size: 10.5pt; color: #111; flex: 1; }
.tag { font-size: 8pt; font-weight: 700; padding: 1.5pt 7pt; white-space: nowrap; }
.tag.top { background: #111; color: #fff; }
.tag.urgent { border: 1.2px solid #111; color: #111; }
.tag.norm { color: #555; border: 1px solid #aaa; }
.dcard .db { padding: 7pt 12pt 8pt; font-size: 9.5pt; }
.dcard .db p { margin: 4pt 0; }
.answer { margin-top: 6pt; padding-top: 5pt; border-top: 1px dashed #999; font-size: 9pt; color: #333; }
.answer .fld { display: inline-block; min-width: 96pt; border-bottom: 1px solid #555; margin: 0 10pt 3pt 3pt; }

/* ---------- 진행 계획 ---------- */
.phases { display: flex; gap: 8pt; margin: 12pt 0; }
.phase { flex: 1; border: 1px solid #999; padding: 8pt 10pt; page-break-inside: avoid; }
.phase .ph { font-weight: 800; color: #111; border-bottom: 1px solid #111; padding-bottom: 3pt; margin-bottom: 5pt; }
.phase ul { margin: 3pt 0 0; padding-left: 14pt; font-size: 8.8pt; }

/* ---------- 도표: 퍼널·분할 도해 ---------- */
.funnel { margin: 8pt 0 12pt; }
.fstep { border: 1px solid #111; padding: 3pt 9pt; margin: 0 0 2.5pt; font-size: 8.8pt;
         background: #f5f5f5; white-space: nowrap; }
.fstep.fdark { background: #111; color: #fff; }
.split-demo { border: 1px solid #999; padding: 6pt 10pt; margin: 7pt 0 2pt; font-size: 8.6pt; }
.sd-row { margin: 3pt 0; }
.sd-label { display: inline-block; min-width: 118pt; font-weight: 700; }
.sd-g { margin-right: 8pt; white-space: nowrap; }
.fr { display: inline-block; border: 1px solid #555; padding: 0.5pt 5pt; margin: 0 1pt; font-size: 8pt; }
.fr.t { background: #e8e8e8; }
.fr.e { background: #111; color: #fff; }
.sd-note { color: #555; }
.sd-bad .sd-label { text-decoration: line-through; text-decoration-thickness: 1px; }

/* ---------- 부록 ---------- */
.appendix { page-break-before: always; }
.appendix h1 { font-size: 16pt; padding-bottom: 6pt; border-bottom: 2.5px solid #111; }
.appendix h2 { font-size: 12.5pt; margin: 16pt 0 7pt; padding-bottom: 3pt; border-bottom: 1px solid #111; }
.small { font-size: 8.7pt; color: #555; }
"""

P1 = """
<div class="cover">
  <div class="kicker">프로젝트 종합 기획서</div>
  <h1>부대 CCTV 클립 →<br>YOLO 학습 데이터 파이프라인</h1>
  <p class="sub">클립 수집부터 모델 재배포까지 — 전 과정 설계 종합안</p>
  <table class="cover-meta">
    <tr><th>문서 상태</th><td>초안 v1 (2026-08-27)</td></tr>
    <tr><th>작성</th><td>학습체계 담당 (CCTV 프로그램·학습 플랫폼 개발)</td></tr>
    <tr><th>구성</th><td>1 개요 · 2 전체 구조 · 3 처리 단계 · 4 확정 사항 · 5 규모·진행·리스크 · 6 미확정 항목 · 부록 A 클립 전달 규격</td></tr>
    <tr><th>상세 문서</th><td>저장소 docs/clip-pipeline/ — README.md(설계 상세) · interface-spec.md(규격) · decisions.md(항목별 배경)</td></tr>
  </table>
</div>

<!-- ================= 1. 개요 ================= -->
<div class="sec" id="s1">
<h2><span class="no">1</span>개요</h2>

<p><b>목적.</b> 부대 CCTV 프로그램이 생성하는 이벤트 클립(기본 15초 = 트리거 전 5초 + 후 10초)을
수집해 탐지 모델을 재학습하고, 개선된 모델을 부대에 재배포하는 순환 체계를 구축한다.
현장에서 모델이 놓친 장면이 우선적으로 학습에 반영된다.</p>

<p><b>배경.</b> 현행은 수동 방식이다 — VLC 로 프레임 추출, 수동 라벨링. 월 배치 수신이 시작되면
월 27,000클립(프레임 약 47만 장) 규모로 수동 처리가 불가하다.</p>

<p><b>방식.</b> 기계가 할 수 있는 일(이미지화·중복 정제·초벌 라벨)은 전량 자동화하고,
사람 검수는 가치 순으로 선별한다. 목적은 전량 라벨링이 아니라 모델 개선이다.</p>

<p><b>개발 범위.</b> CCTV 프로그램, 부대→서버 전송 체계, 학습 플랫폼 — 소프트웨어 전 구간을
자체 개발한다. 외부 도입은 네트워크 회선과 서버 하드웨어뿐이다. 데이터 표준·인력·정책 등
미확정 항목은 6절에 정리한다.</p>

<div class="stats">
  <div class="stat"><div class="v">27,000</div><div class="k">월 유입 클립<br>(3개 부대 합산)</div></div>
  <div class="stat"><div class="v">47만 → 19만</div><div class="k">월 프레임<br>자동 정제 후 처리량</div></div>
  <div class="stat"><div class="v">2명 × 5h</div><div class="k">검수 인력 산정 기준<br>(1일 기준)</div></div>
  <div class="stat"><div class="v">H200×16 · 21TB</div><div class="k">도입 확정 GPU·스토리지</div></div>
</div>
</div>

<!-- ================= 2. 전체 구조 ================= -->
<div class="sec" id="s2">
<h2><span class="no">2</span>전체 구조</h2>

<div class="flow">
  <div class="zone">
    <span class="zt">① 부대 CCTV 감시망</span> <span class="zs">(분리망 — 직접 전송 불가)</span>
    <div class="steps"><span class="step">이벤트 감지 → 클립 저장 (트리거 전 5초 + 후 10초)</span>
    <span class="step">클립별 메타데이터 파일(사이드카) 생성</span>
    <span class="step">배치 내보내기 — 외장하드에 클립+목록+해시 기록</span></div>
  </div>
  <div class="flow-arrow"><b>▼ 외장하드 물리 이동 (월 1회 — 부대당 약 63GB)</b></div>
  <div class="zone hl">
    <span class="zt">② 학습 플랫폼 (GPU 서버)</span>
    <div class="steps">
      <span class="step">1. 업로드·검증 — 해시 대조, 규격 위반 격리</span>
      <span class="step">2. 이미지화 — 1초 간격(설정), 무변화 프레임 미저장</span>
      <span class="step">3. 초벌 라벨 — 전량 자동 (19만 장, 수십 분)</span>
      <span class="step">4. 검수 — 미탐 후보 우선, 예산제</span>
      <span class="step">5. 데이터셋 편성 — 누수 방지 분할, 버전 고정</span>
      <span class="step">6. 학습 → 부대별 평가 → 합부 판정</span>
    </div>
  </div>
  <div class="flow-arrow"><b>▼ 합격 모델 배포</b> (모델 + 클래스 대응표) &nbsp;&nbsp;|&nbsp;&nbsp; 원본 클립 ▶ 데이터관리포탈 이관 후 서버에서 삭제</div>
  <div class="zone">
    <span class="zt">③ 부대 재배포</span>
    <div class="steps"><span class="step">CCTV 프로그램이 현장 GPU 에 맞는 엔진 생성</span>
    <span class="step">다음 배치부터 모델 세대별 미탐 추적 가능</span></div>
  </div>
</div>

<p><b>설계 원칙.</b>
① 기계는 전량, 사람은 예산만큼.
② 부대가 저장·평가·처방의 기본 단위 — 모델은 공통, 데이터 배합과 지표만 부대별.
③ 같은 사건의 프레임은 통째로 한 분할에만(성능 부풀림 방지).
④ 규격 위반 데이터는 폐기하지 않고 격리·회신.
⑤ 수치 가정은 명시하고 시범 배치 1회로 실측 보정.</p>
</div>

<!-- ================= 3. 처리 단계 ================= -->
<div class="sec" id="s3">
<h2><span class="no">3</span>처리 단계</h2>

<div class="stage"><div class="sh"><span class="n">1</span>수집·이동</div>
<p>CCTV 프로그램의 배치 내보내기가 클립·사이드카·매니페스트(해시 포함)를 외장하드에 기록한다.
월 1회 물리 이동 후 업로드 도구로 서버에 적재한다. 수신 시 해시를 대조해 손상·누락을 자동
검출하고, 규격 위반은 격리 후 배치별 준수율 리포트를 생성한다 — 격리 원인 대부분은 부대 장비
상태(시계 미동기 등) 신호다.</p></div>

<div class="stage"><div class="sh"><span class="n">2</span>이미지화</div>
<p>클립을 1초 간격(부대별 설정 가능)으로 추출한다. 무변화 프레임은 저장하지 않는다
(월 47만 → 약 19만 장). 연속 클립이 같은 시간대를 겹쳐 담는 경우 시각 기준으로 1회만 추출한다.</p></div>

<div class="stage"><div class="sh"><span class="n">3</span>초벌 라벨 (자동)</div>
<p>최신 승인 모델로 전량 추론한다(H200 기준 수십 분). 결과는 신뢰도로 3분류한다 —
자동승인(5% 표본 감사) / 검수 대상(경계 신뢰도) / 무검출(미탐 후보). 클립은 이벤트 감지
시에만 생성되므로 무검출 프레임은 모델의 약점 후보이며 검수 1순위다.</p></div>

<div class="stage"><div class="sh"><span class="n">4</span>검수</div>
<p>전수 검수는 월 293시간(상근 3명 상당)으로 불가하다. 검수 시간을 예산으로 두고
미탐 후보 전량 → 표본 감사 → 경계 케이스 순으로 소진한다. 산정 기준 2명×5h/일.
검수는 웹 화면에서 클립 단위로 박스를 확인·수정하며, 전 행위가 기록된다.</p></div>

<div class="stage"><div class="sh"><span class="n">5</span>데이터셋 편성</div>
<p>승인분만 편입한다. 같은 사건의 프레임은 동일 분할(train/val/test)에만 배정한다 — 1초 간격
프레임은 상호 근접 중복이라 무작위 분할 시 성능이 과대평가된다. test 는 영구 고정,
배포 판정용 골든셋(부대별 500~1,000장, 사람 전수 라벨)은 별도 유지한다. 축적분과 유사한
반복 장면은 대표만 편입한다.</p>
<div class="split-demo">
  <div class="sd-row sd-bad"><span class="sd-label">프레임 무작위 분할 (금지)</span>
    <span class="sd-g">사건 A: <span class="fr t">학습</span><span class="fr e">평가</span><span class="fr t">학습</span><span class="fr e">평가</span></span>
    <span class="sd-note">→ 거의 같은 프레임이 학습·평가에 나뉘어 성능 과대평가</span></div>
  <div class="sd-row"><span class="sd-label">사건 그룹 분할 (채택)</span>
    <span class="sd-g">사건 A: <span class="fr t">학습</span><span class="fr t">학습</span><span class="fr t">학습</span><span class="fr t">학습</span></span>
    <span class="sd-g">사건 B: <span class="fr e">평가</span><span class="fr e">평가</span><span class="fr e">평가</span></span>
    <span class="sd-note">→ 사건 단위로만 배정</span></div>
</div></div>

<div class="stage"><div class="sh"><span class="n">6</span>학습·평가</div>
<p>학습은 운용 중인 학습 콘솔로 수행한다. 평가는 부대별×클래스별 지표로 집계한다 — 전체
평균은 특정 부대의 저성능을 가린다. 합부 기준(필수 클래스·부대별 최저선, 직전 모델 대비
하락 한도, test·골든셋 동시 통과)을 만족한 모델만 배포하고 판정 근거를 이력으로 남긴다.
저성능 부대는 다음 사이클에서 해당 부대 데이터의 검수·편입 비중을 높인다.</p></div>

<div class="stage"><div class="sh"><span class="n">7</span>배포·원본 정리</div>
<p>모델은 ONNX + 클래스 대응표 묶음으로 배포하고, CCTV 프로그램이 현장 GPU 에 맞는 엔진을
생성한다(부대별 GPU 상이 무관). 원본 클립은 추출 완료 후 데이터관리포탈로 이관하고 서버에서
삭제한다 — 서버에는 이미지와 메타데이터만 남는다.</p></div>
</div>

<!-- ================= 4. 확정 사항 ================= -->
<div class="sec" id="s4">
<h2><span class="no">4</span>확정 사항</h2>
<table>
<tr><th>영역</th><th>내용</th></tr>
<tr><td>개발 주체</td><td>CCTV 프로그램·전송 체계·학습 플랫폼 전부 자체 개발. 외부 도입은 네트워크 회선·서버 하드웨어</td></tr>
<tr><td>인프라</td><td>리눅스 GPU 서버 H200 × 16장 + 스토리지 21TB · 쿠버네티스 · Docker 이미지 반입 · 서버 내부 구성(DB 포함) 자체 설계</td></tr>
<tr><td>런타임</td><td>Python 3.14.7 고정(보안 지침) · 웹 방식 · 기존 학습 콘솔 코어 확장</td></tr>
<tr><td>클래스</td><td>표출: 사람/차량/동물 (+해안 부대: 튜브/보트) 고정 — 현행 화면 표시와 동일. 학습 내부 세분만 별도 확정(D-07)</td></tr>
<tr><td>전송</td><td>감시망은 분리망 — 월 1회 외장하드 반출 → 업로드. 내보내기·업로드 도구 자체 개발</td></tr>
<tr><td>원본 처분</td><td>추출 후 데이터관리포탈 이관, 서버에서 삭제. 이미지·메타데이터만 보관</td></tr>
<tr><td>초기 데이터</td><td>기존 보유 약 5만 장 정리(중복 제거·라벨·검수) → 부대별 골든셋 선별 → 베이스 모델 재학습. 파이프라인 가동 전 선행</td></tr>
<tr><td>모델 배포 형식</td><td>ONNX 정본 + 클래스 대응표. TensorRT 엔진은 현장 생성(엔진은 GPU 종속)</td></tr>
</table>
</div>

<!-- ================= 5. 규모·진행·리스크 ================= -->
<div class="sec" id="s5">
<h2><span class="no">5</span>규모 산정 · 진행 계획 · 리스크</h2>

<h3>월 배치 1회 기준 처리 규모</h3>
<div class="funnel">
  <div class="fstep" style="width:100%"><b>원시 프레임 47만 장</b> — 27,000클립 × 1초 간격</div>
  <div class="fstep" style="width:64%"><b>19만 장</b> — 무변화 프레임 자동 제거 (60% 가정)</div>
  <div class="fstep" style="width:44%"><b>검수 후보 8.2만 장</b> — 신뢰도 분류</div>
  <div class="fstep fdark" style="width:30%"><b>사람 검수</b> — 예산 선별</div>
</div>
<table>
<tr><th>구간</th><th>규모</th><th>비고</th></tr>
<tr><td>수신 클립</td><td>27,000개 · 약 190GB</td><td>3개 부대 × 300개/일 × 30일. 부대 확장 시 선형 증가</td></tr>
<tr><td>프레임 (정제 후)</td><td>약 19만 장 · 약 85GB/월</td><td>무변화 제거율 60% 가정 — 시범 배치로 실측 보정</td></tr>
<tr><td>초벌 라벨</td><td>19만 장 전량</td><td>H200 기준 수십 분</td></tr>
<tr><td>검수</td><td>후보 최대 8.2만 장 중 예산만큼 선별</td><td>2명×5h/일 → 미탐 전량 + 감사 + 경계 케이스 절반</td></tr>
<tr><td>저장 (서버)</td><td>이미지 연 약 1TB</td><td>21TB 대비 수년 여유</td></tr>
</table>
<p class="small">수치는 가정 명시된 추정치. 조정 레버(추출 간격·신뢰도 구간·검수 예산)는 전부 설정값.</p>

<h3>진행 계획</h3>
<div class="phases">
  <div class="phase"><div class="ph">1단계 — 기반</div>
    <ul><li>기존 5만 장 정리 (중복 제거·분류)</li>
    <li>학습 클래스 확정(D-07) 후 라벨·검수</li>
    <li>부대별 골든셋 구축</li>
    <li>베이스 모델 재학습</li>
    <li>Python 3.14.7 런타임 이행</li></ul></div>
  <div class="phase"><div class="ph">2단계 — 시범</div>
    <ul><li>전달 규격(부록 A)의 CCTV 프로그램 구현</li>
    <li>수신·추출·초벌 라벨 파이프라인 개발</li>
    <li>시범 배치 1회 — 가정 실측·보정</li>
    <li>검수 화면 개발 (병행)</li></ul></div>
  <div class="phase"><div class="ph">3단계 — 정규 운영</div>
    <ul><li>GPU 서버 이행(Docker)·계정 체계</li>
    <li>월 사이클: 수신→처리→검수→학습→배포</li>
    <li>부대별 지표·처방 루프 가동</li>
    <li>부대 확장 시 온보딩 절차 적용</li></ul></div>
</div>

<h3>주요 리스크</h3>
<table>
<tr><th>리스크</th><th>대응 (설계 반영)</th></tr>
<tr><td>모델이 놓친 것은 라벨도 안 되는 자기강화</td><td>이벤트 발생-무검출 프레임 최우선 검수 · 자동승인 표본 감사 · 골든셋은 사람 라벨</td></tr>
<tr><td>야간·악천후 데이터 편중</td><td>분포 리포트 상시 확인 · 검수 샘플링 야간 쿼터 · 주/야 분리 지표</td></tr>
<tr><td>고정 카메라 반복 장면으로 데이터 중복</td><td>축적분 대비 신규성 선별 편입 · 카메라별 상한</td></tr>
<tr><td>검수 인력 병목</td><td>예산제 — 파이프라인은 정지하지 않음. 미검수분 보류 보관, 우선순위 조정</td></tr>
<tr><td>보안 (영상 취급)</td><td>부대 단위 접근·격리 · 검수 행위 전수 기록 · 서버 이행 시 인증 · 보존·파기 정책(D-05)</td></tr>
</table>
</div>

<!-- ================= 6. 미확정 항목 ================= -->
<div class="sec" id="s6">
<h2><span class="no">6</span>미확정 · 확인 필요 항목</h2>
<p>아래 항목은 외부 확인·결정이 필요해 미확정 상태다. 번호(D-nn)는 상세 문서(decisions.md)와
공통이다.</p>

<table>
<tr><th style="width:52pt">번호</th><th style="width:110pt">항목</th><th>확인·결정할 내용</th></tr>
<tr id="d-07"><td>D-07</td><td>학습 클래스 세분</td><td>부대별 실제 출현 객체(동물·차량 종류), 통계상 구분 수요. 초안: 차량 → 일반/군용/미상, 동물 → 멧돼지/고라니/조류/기타. <b>기존 5만 장 라벨 작업의 선행 조건</b></td></tr>
<tr id="d-01"><td>D-01</td><td>데이터 표준</td><td>부록 A 규격(파일명·코드 체계·메타데이터)의 표준 확정. 데이터과 자체 명명·분류 표준 유무. 보유 현황 정기 보고 필요 여부. 배치 주기(월 1회) 확인</td></tr>
<tr id="d-02"><td>D-02</td><td>반출입 절차</td><td>외장하드 매체 지정·반출입 승인·운반 담당·업로드 후 초기화 규정. 준수율 리포트 회람 경로</td></tr>
<tr id="d-03"><td>D-03</td><td>검수 인력</td><td>담당 인원·소속·1일 가용 시간. 산정 기준 2명×5h — 1명이면 미탐 확인 위주 축소, 3명이면 잔여 해소</td></tr>
<tr id="d-04"><td>D-04</td><td>골든셋 인력</td><td>부대당 500~1,000장 사람 전수 라벨. 초기 3개 부대 약 5~10인일. D-03 인력 겸임 가능</td></tr>
<tr id="d-05"><td>D-05</td><td>포탈 연동·보존</td><td>이관 방식(경로·계정·해시 검증). 클립 재수령(다운로드) 가능 여부 — 재추출의 유일 경로. 포탈 보존 연한. 서버 이미지 보존·파기 절차. 격리 클립 처리</td></tr>
<tr id="d-06"><td>D-06</td><td>마스킹 여부</td><td>얼굴·번호판 마스킹 요구 여부(보안 판단). 마스킹은 탐지 학습 품질과 상충 — 미적용 시 접근통제·행위 기록으로 갈음</td></tr>
<tr id="d-08"><td>D-08</td><td>서버 운영</td><td>도입 시점. 학습 플랫폼 GPU 할당(상시 수요 1~2장). 드라이버 550+ 설치. k8s 운영 주체. 로그인 인증 방식(기존 계정 연동/자체). 업로드 지점 환경. 검수 PC 표준 브라우저</td></tr>
<tr id="d-09"><td>D-09</td><td>Docker 반입</td><td>이미지(약 10GB) 심사 방식 — SBOM·다이제스트 요구 여부. 내부 레지스트리 유무. 코드 레이어 단위 반입 허용 여부</td></tr>
<tr id="d-10"><td>D-10</td><td>부대 실사</td><td>부대별 카메라 대수·배치. 화각 중첩 카메라 쌍. 야간 클립 비중. CCTV 처리 PC 의 GPU 기종</td></tr>
<tr id="d-11"><td>D-11</td><td>배포 절차</td><td>모델 전달 경로(클립 반입 역방향)·적용 주기·시범 부대. 안: 월 사이클, 초회 1개 부대 시범</td></tr>
<tr id="d-12"><td>D-12</td><td>합부 기준 수치</td><td>배포 판정 수치(클래스·부대별 최저선, 허용 하락폭)의 승인 주체·확정 시점. 초안은 시범 배치 실측 후 작성</td></tr>
</table>

<p class="small">내부 확정(외부 확인 불요): CCTV 프로그램의 규격 구현(사이드카·해시·사건 ID·오탐/미탐
신고 코드·배치 내보내기·현장 엔진 생성), 학습 플랫폼 전체 구현, 코드 대장(부대·카메라·이벤트·클래스)
관리, 구버전 프로그램 과도기 대응(격리·회신).</p>
</div>
"""

APPENDIX_INTRO = """
<div class="appendix" id="appendix-a">
<p class="small"><b>부록 A</b> — 클립 전달 규격 원문(구현 기준, D-01 확정 대상).
본문 절 번호 인용은 저장소 상세 설계 문서(README.md) 기준.</p>
"""

html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>부대 CCTV 클립 → YOLO 학습 데이터 파이프라인 기획서</title>
<style>{CSS}</style></head>
<body>
{P1}
{APPENDIX_INTRO}
{spec_html}
</div>
</body></html>
"""

OUT.write_text(html, encoding="utf-8")
print("written:", OUT, len(html), "chars")
