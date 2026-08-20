import { useEffect, useState } from 'react'

import { api } from '../api'
import type {
  AnalysisBox,
  AnalysisReport,
  Dataset,
  JobStatus,
  LabelIssues,
  NextAction,
  Run,
  TideBreakdown,
} from '../types'
import { BoxOverlay, type OverlayBox } from './BoxOverlay'
import { DatasetPathWarning } from './DatasetPathWarning'
import { Modal } from './ui/Dialog'
import { EmptyState } from './ui/EmptyState'

const POLL_MS = 2000

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? '-' : value.toFixed(3)
}

/** 색은 BoxOverlay 가 정한다. 여기서는 무엇인지만 말한다. */
function toOverlay(boxes: AnalysisBox[], kind: 'gt' | 'pred'): OverlayBox[] {
  return boxes.map((b) => {
    if (kind === 'gt') {
      const missed = b.state === 'miss'
      return {
        box: b.box,
        label: missed ? `놓침: ${b.name}` : `정답: ${b.name}`,
        kind: missed ? 'missed' : 'gt',
      }
    }
    const wrong = b.state === 'false'
    return {
      box: b.box,
      label: `${wrong ? '오검출' : '검출'}: ${b.name} ${b.conf ?? ''}`,
      kind: wrong ? 'false' : 'hit',
    }
  })
}

/**
 * 표를 읽기 전에 결론부터.
 *
 * 비전문가가 실제로 읽고 따라 하는 것은 이 카드다. 그래서 맨 위에 둔다. 판정도 문장도
 * 서버가 만든 것을 그대로 쓴다.
 */
function NextActionCard({ actions }: { actions: NextAction[] }) {
  return (
    <div className="card card-lead">
      <div className="card-head">
        <h3>다음에 할 일</h3>
      </div>
      {actions.map((action) => (
        <div key={action.code} style={{ marginTop: 'var(--sp-3)' }}>
          <strong className={action.severity === 'critical' ? 'error' : undefined}>
            {action.title}
          </strong>
          <p className="small">{action.cause}</p>
          <p className="small muted">{action.fix}</p>
        </div>
      ))}
    </div>
  )
}

/**
 * 모델이 아니라 라벨이 틀렸을 법한 자리.
 *
 * 의심 지점만 굵게 그리고 나머지 예측은 그리지 않는다. 다 그리면 무엇을 보라는 것인지
 * 알 수 없어진다.
 */
function LabelIssueCard({
  issues,
  datasetId,
  onZoom,
}: {
  issues: LabelIssues
  datasetId: string
  onZoom: (src: string) => void
}) {
  return (
    <div className="card">
      <div className="card-head">
        <h3>라벨 오류 후보</h3>
        <span className="small muted spacer">
          {issues.total}건 중 {issues.shown}건
        </span>
      </div>
      {/* val 한정이라는 사실이 사진보다 먼저 눈에 들어와야 한다. */}
      <p className="help warn">{issues.scope_note}</p>
      {issues.reason && <p className="help warn">{issues.reason}</p>}
      {issues.kinds.length > 0 && (
        <p className="small muted">
          {issues.kinds.map((k) => `${k.label} ${k.count}`).join(' · ')}
        </p>
      )}
      {issues.items.length === 0 ? (
        <EmptyState title="라벨 오류로 의심할 만한 자리를 찾지 못했습니다." />
      ) : (
        <div className="gallery">
          {issues.items.map((item) => (
            <figure key={item.image}>
              <BoxOverlay
                src={api.datasetImageUrl(datasetId, item.image)}
                alt={item.name}
                boxes={[
                  ...item.gt.map((b) => ({
                    box: b.box,
                    label: `정답: ${b.name}`,
                    kind: 'gt' as const,
                  })),
                  ...item.findings.flatMap((f) =>
                    f.ref_box
                      ? [{ box: f.ref_box, label: `근거: ${f.ref_name}`, kind: 'evidence' as const }]
                      : [],
                  ),
                  ...item.findings.map((f) => ({
                    box: f.box,
                    label: `${f.label}: ${f.name}`,
                    kind: 'missed' as const,
                  })),
                ]}
                onZoom={onZoom}
              />
              <figcaption>
                {item.name}
                {item.findings.map((f, i) => (
                  <span key={i} className="muted">
                    <br />
                    {f.message}
                  </span>
                ))}
              </figcaption>
            </figure>
          ))}
        </div>
      )}
      <p className="help">주황 굵게=의심 지점 · 회색 점선=근거 박스 · 초록=나머지 정답</p>
    </div>
  )
}

/**
 * 오류 유형별로 mAP 를 얼마나 깎는지.
 *
 * 숫자만 늘어놓으면 비전문가는 여전히 다음 행동을 못 정하므로, 손실이 큰 쪽 셋의 처방
 * 문장을 함께 싣는다. 문장은 서버가 만든 것을 그대로 쓴다.
 */
function TideCard({ tide }: { tide: TideBreakdown }) {
  const rows = [...tide.errors].sort((a, b) => (b.dap ?? 0) - (a.dap ?? 0))
  const worst = Math.max(...rows.map((r) => r.dap ?? 0), 0)
  // 상승분이 미미한 유형까지 처방을 띄우면 mAP50 0.95 짜리 모델에도 "학습이 부족합니다"
  // 가 뜬다. 고쳐서 얻을 게 있는지는 서버가 판정한다.
  const advice = rows.filter((r) => r.actionable).slice(0, 3)

  return (
    <div className="card">
      <div className="card-head">
        <h3>무엇이 mAP 를 깎는가</h3>
        <span className="small muted spacer">
          이 분석 기준 mAP50 {pct(tide.baseline_map50)}
        </span>
      </div>
      <table>
        <thead>
          <tr>
            <th>오류 유형</th>
            <th style={{ width: '40%' }}>mAP50 손실</th>
            <th>손실</th>
            <th>건수</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.kind}>
              <td>{row.label}</td>
              <td>
                <div className="progress">
                  <div style={{ width: worst > 0 ? `${((row.dap ?? 0) / worst) * 100}%` : '0%' }} />
                </div>
              </td>
              <td>{pct(row.dap)}</td>
              {/* 전체 검출 기준 건수를 앞에 두면 conf 0.001 짜리 잡음이 수백 건으로
                  읽혀 멀쩡한 모델을 고치러 가게 된다. 배포 임계값에서 보이는 수가 먼저다. */}
              <td>
                {row.count_at_conf ?? row.count}
                {row.count_at_conf != null && row.count_at_conf !== row.count && (
                  <span className="muted"> ({row.count})</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {advice.length === 0 ? (
        <p className="help" style={{ marginTop: 'var(--sp-3)' }}>
          어느 유형을 고쳐도 mAP50 이 의미 있게 오르지 않습니다. 특정 오류를 손볼 단계가
          아니니 위 '다음에 할 일' 을 보세요.
        </p>
      ) : (
        advice.map((row) => (
          <p key={row.kind} className="help" style={{ marginTop: 'var(--sp-3)' }}>
            <strong>{row.label}</strong> {row.advice}
          </p>
        ))
      )}
      <p className="small muted" style={{ marginTop: 'var(--sp-3)' }}>
        {tide.note}
      </p>
    </div>
  )
}

/**
 * 학습 결과 진단 — 클래스별 성능, 배포용 신뢰도, 실제로 틀린 사진.
 *
 * 분석은 검증을 한 번 더 도는 별도 프로세스라 시간이 걸린다. 잡으로 띄우고 폴링한다
 * (내보내기와 같은 방식).
 */
export function DiagnosePanel({ run, dataset }: { run: Run; dataset?: Dataset | null }) {
  const [job, setJob] = useState<JobStatus | null>(null)
  const [report, setReport] = useState<AnalysisReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [zoom, setZoom] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setReport(null)
    setJob(null)
    api.analysisStatus(run.id).then((s) => !cancelled && setJob(s)).catch(() => {})
    api.analysisReport(run.id).then((r) => !cancelled && setReport(r)).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [run.id])

  // 도는 동안만 폴링한다. 끝나면 리포트를 한 번 더 받아 온다.
  useEffect(() => {
    if (job?.status !== 'running') return
    let cancelled = false
    const timer = setInterval(async () => {
      try {
        const next = await api.analysisStatus(run.id)
        if (cancelled) return
        setJob(next)
        if (next.status === 'completed') {
          setReport(await api.analysisReport(run.id))
        }
      } catch {
        /* 폴링 실패는 다음 주기에 다시 시도한다 */
      }
    }, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [job?.status, run.id])

  const start = async () => {
    setBusy(true)
    setError('')
    try {
      setJob(await api.startAnalysis(run.id, { imgsz: 640, batch: 8, use_gpu: false }))
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  if (run.status !== 'completed' && run.status !== 'stopped') {
    return <EmptyState title="학습이 끝난 뒤에 진단할 수 있습니다." />
  }

  const running = job?.status === 'running'
  const events = job?.events ?? []
  const stage = events.length ? events[events.length - 1].message : undefined

  return (
    <div className="stack">
      {/* 경로가 낡았으면 갤러리 사진이 전부 깨진다. 사유가 없으면 "모델이 다 놓쳤다" 로 읽힌다. */}
      <DatasetPathWarning dataset={dataset} />

      <div className="card">
        <div className="card-head">
          <h3>오류 분석</h3>
          <button className="btn-sm spacer" onClick={start} disabled={busy || running}>
            {running ? '분석 중…' : report ? '다시 분석' : '분석 시작'}
          </button>
        </div>
        <p className="help">
          검증 셋을 한 번 더 돌려 클래스별 성능과 실제로 틀린 사진을 모읍니다. CPU 로 도므로
          학습 중이어도 GPU 를 뺏지 않습니다.
        </p>
        {running && <p className="small muted">{stage ?? '준비 중…'}</p>}
        {error && <p className="error small">{error}</p>}
        {job?.status === 'failed' && (
          <p className="error small">
            분석이 실패했습니다: {String(job.result?.error ?? job.error ?? '')}
          </p>
        )}
      </div>

      {report && (
        /*
         * 2열. 왼쪽은 "무엇을 할까" 로 읽는 흐름(다음 행동 → 무엇이 깎는가 → 실제 사진),
         * 오른쪽은 참고용 수치다. 한 줄로 쌓으면 결론이 스크롤 아래로 밀린다.
         */
        <div className="diag-grid">
          <div className="stack">
            {report.next_actions && report.next_actions.length > 0 && (
              <NextActionCard actions={report.next_actions} />
            )}

            {/* 키 자체가 없으면 오류 분해가 생기기 전(schema_version 1)의 리포트다. */}
            {report.tide?.failed && <p className="help warn">{report.tide.message}</p>}
            {report.tide && !report.tide.failed && <TideCard tide={report.tide} />}

            <div className="card">
              <div className="card-head">
                <h3>가장 많이 틀린 사진</h3>
                <span className="small muted spacer">
                  {report.gallery_total}장 중 {report.gallery.length}장 · 신뢰도{' '}
                  {report.gallery_conf} 기준
                </span>
              </div>
              {report.gallery.length === 0 ? (
                <EmptyState title="틀린 사진이 없습니다." />
              ) : (
                <div className="gallery">
                  {report.gallery.map((item) => (
                    <figure key={item.image}>
                      <BoxOverlay
                        src={api.datasetImageUrl(run.dataset_id, item.image)}
                        alt={item.name}
                        boxes={[...toOverlay(item.gt, 'gt'), ...toOverlay(item.pred, 'pred')]}
                        onZoom={setZoom}
                      />
                      <figcaption>
                        {item.name}
                        <br />
                        <span className="muted">
                          놓침 {item.fn} · 오검출 {item.fp} · 맞음 {item.tp}
                        </span>
                      </figcaption>
                    </figure>
                  ))}
                </div>
              )}
              <p className="help">
                초록=정답 · 주황=놓친 정답 · 파랑=맞은 검출 · 빨강 점선=오검출
              </p>
            </div>
          </div>
          <div className="stack">
            <div className="card">
              <div className="card-head">
                <h3>클래스별 성능</h3>
                <span className="small muted spacer">
                  이미지 {report.overall.images} · 인스턴스 {report.overall.instances} · mAP50-95{' '}
                  {pct(report.overall.map50_95)}
                </span>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>클래스</th>
                    <th>인스턴스</th>
                    <th>정밀도</th>
                    <th>재현율</th>
                    <th>mAP50</th>
                    <th>mAP50-95</th>
                  </tr>
                </thead>
                <tbody>
                  {report.per_class.map((c) => (
                    <tr key={c.cls}>
                      <td>{c.name}</td>
                      <td>{c.instances}</td>
                      <td>{pct(c.precision)}</td>
                      <td>{pct(c.recall)}</td>
                      <td>{pct(c.ap50)}</td>
                      <td className={c.evaluated ? undefined : 'muted'}>
                        {c.evaluated ? pct(c.ap50_95) : '평가 안 됨'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {report.worst_classes.map((w, i) => (
                <div key={i} className="help warn" style={{ marginTop: 'var(--sp-3)' }}>
                  {w.message}
                </div>
              ))}
            </div>

            <div className="card">
              <div className="card-head">
                <h3>배포용 신뢰도 임계값</h3>
              </div>
              {report.conf_recommendation.reliable ? (
                <p className="small">
                  <strong className="conf-value">{report.conf_recommendation.conf}</strong> 에서 F1 이 가장 높습니다 (
                  {pct(report.conf_recommendation.f1)}). 정밀도 {pct(report.conf_recommendation.precision)} ·
                  재현율 {pct(report.conf_recommendation.recall)}.
                  <span className="muted"> 추론 화면 기본값은 0.25 입니다.</span>
                </p>
              ) : (
                <p className="help warn">{report.conf_recommendation.message}</p>
              )}
            </div>

            {report.label_issues && (
              <LabelIssueCard
                issues={report.label_issues}
                datasetId={run.dataset_id}
                onZoom={setZoom}
              />
            )}

          </div>
        </div>
      )}

      <Modal
        open={zoom !== null}
        onClose={() => setZoom(null)}
        className="dialog lightbox"
        label="확대한 이미지"
      >
        {zoom && <img src={zoom} alt="확대한 사진" />}
      </Modal>
    </div>
  )
}
