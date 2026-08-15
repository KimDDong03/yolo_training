import { useEffect, useState } from 'react'

import { api } from '../api'
import type { AnalysisBox, AnalysisReport, JobStatus, Run, TideBreakdown } from '../types'
import { BoxOverlay, type OverlayBox } from './BoxOverlay'
import { Modal } from './ui/Dialog'
import { EmptyState } from './ui/EmptyState'

const POLL_MS = 2000

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? '-' : value.toFixed(3)
}

/** 정답=초록(놓친 것은 주황 굵게), 맞은 예측=파랑, 오검출=빨강 점선. */
function toOverlay(boxes: AnalysisBox[], kind: 'gt' | 'pred'): OverlayBox[] {
  return boxes.map((b) => {
    if (kind === 'gt') {
      const missed = b.state === 'miss'
      return {
        box: b.box,
        label: missed ? `놓침: ${b.name}` : `정답: ${b.name}`,
        color: missed ? 'var(--warn)' : 'var(--ok)',
        emphasis: missed,
      }
    }
    const wrong = b.state === 'false'
    return {
      box: b.box,
      label: `${wrong ? '오검출' : '검출'}: ${b.name} ${b.conf ?? ''}`,
      color: wrong ? 'var(--err)' : 'var(--accent)',
      dashed: wrong,
    }
  })
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
  const advice = rows.filter((r) => (r.dap ?? 0) > 0).slice(0, 3)

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
              <td>{row.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {advice.length === 0 ? (
        <p className="help" style={{ marginTop: 'var(--sp-3)' }}>
          어떤 유형을 고쳐도 mAP50 이 오르지 않습니다. 특정 오류를 손볼 단계가 아니라 모델이
          아직 제대로 학습되지 않은 상태입니다.
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
export function DiagnosePanel({ run }: { run: Run }) {
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
        <>
          {/* 키 자체가 없으면 오류 분해가 생기기 전(schema_version 1)의 리포트다. */}
          {report.tide?.failed && <p className="help warn">{report.tide.message}</p>}
          {report.tide && !report.tide.failed && <TideCard tide={report.tide} />}

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
                <strong>{report.conf_recommendation.conf}</strong> 에서 F1 이 가장 높습니다 (
                {pct(report.conf_recommendation.f1)}). 정밀도 {pct(report.conf_recommendation.precision)} ·
                재현율 {pct(report.conf_recommendation.recall)}.
                <span className="muted"> 추론 화면 기본값은 0.25 입니다.</span>
              </p>
            ) : (
              <p className="help warn">{report.conf_recommendation.message}</p>
            )}
          </div>

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
        </>
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
