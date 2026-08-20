import { useEffect, useState } from 'react'

import { api } from '../api'
import { formatDuration } from '../format'
import type { Dataset, Estimate, Recommendation } from '../types'

/** 값 하나를 사람이 읽는 문자열로. */
function show(value: unknown): string {
  if (value === null || value === undefined || value === '') return '없음'
  if (typeof value === 'boolean') return value ? '켬' : '끔'
  if (value === -1) return '자동'
  return String(value)
}

/**
 * 데이터셋 성격에 따른 파라미터 제안과, 이 설정으로 걸릴 시간·VRAM.
 *
 * 이 조회가 컴포넌트가 아니라 훅인 이유 — 목표 카드(간편 모드)도 같은 추천값을 읽어야 한다.
 * 카드가 따로 부르면 같은 질문을 두 번 하게 되고, 두 화면이 서로 다른 값을 보일 수 있다.
 * 부모가 한 번 부르고 아래로 내린다.
 *
 * 폼 값이 바뀔 때마다 서버에 물어보므로 디바운스한다. 모델 경로 검증(NewRunPanel)과 같은
 * cancelled 패턴을 쓴다 — 타이머만 취소하면 이미 날아간 이전 요청의 응답이 늦게 도착해
 * 지금 화면과 어긋난 값을 덮어쓴다.
 */
export function useAdvice(
  dataset: Dataset | undefined,
  values: Record<string, unknown>,
  devices: number[],
): { rec: Recommendation | null; est: Estimate | null } {
  const [rec, setRec] = useState<Recommendation | null>(null)
  const [est, setEst] = useState<Estimate | null>(null)

  // 제안·추정에 실제로 영향을 주는 값만 의존성으로 삼는다. 폼 전체를 넣으면
  // 관계없는 필드를 건드릴 때마다 요청이 나간다.
  const signature = JSON.stringify([
    dataset?.id,
    devices,
    values['model'],
    values['imgsz'],
    values['epochs'],
    values['batch'],
    values['amp'],
    values['patience'],
    values['mixup'],
    values['cache'],
    values['close_mosaic'],
  ])

  useEffect(() => {
    if (!dataset) {
      setRec(null)
      setEst(null)
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      api
        .recommendation(dataset.id, values, devices)
        .then((r) => !cancelled && setRec(r))
        .catch(() => !cancelled && setRec(null))
      api
        .estimate(dataset.id, values, devices)
        .then((e) => !cancelled && setEst(e))
        .catch(() => !cancelled && setEst(null))
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
    // signature 가 바뀔 때만 다시 묻는다. values 를 그대로 넣으면 매 입력마다 돈다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature])

  return { rec, est }
}

/** 전문 모드에서 제안 표와 추정 시간을 펼쳐 보여준다. 조회는 부모(useAdvice)가 한다. */
export function Recommendations({
  values,
  rec,
  est,
  onApply,
}: {
  values: Record<string, unknown>
  rec: Recommendation | null
  est: Estimate | null
  onApply: (patch: Record<string, unknown>) => void
}) {
  const [showAssumptions, setShowAssumptions] = useState(false)

  const items = rec?.items ?? []
  const advisories = rec?.advisories ?? []
  const hasEstimate = est?.ok === true
  if (!items.length && !advisories.length && !hasEstimate) return null

  return (
    <div className="card">
      <div className="card-head">
        <h3>이 데이터셋에 맞는 설정</h3>
      </div>

      {hasEstimate && est && (
        <div className="stack" style={{ gap: 4, marginBottom: items.length ? 'var(--sp-4)' : 0 }}>
          <div className="row tight wrap small">
            <strong>예상 {formatDuration(est.total_time_s)}</strong>
            <span className="muted">
              (에폭당 {formatDuration(est.epoch_time_s)} × {show(values['epochs'])}에폭)
            </span>
            {est.vram_gb !== null && est.vram_total_gb !== null && (
              <span className={est.vram_level === 'ok' ? 'muted' : 'error'}>
                · VRAM {est.vram_gb} / {est.vram_total_gb} GB
              </span>
            )}
            <button className="btn-xs spacer" onClick={() => setShowAssumptions((v) => !v)}>
              {showAssumptions ? '가정 접기' : '가정 보기'}
            </button>
          </div>
          {/* 점 추정만 보여주면 안 된다. 보정 표본이 없을 때는 특히 크게 빗나간다. */}
          <div className="help">
            {est.source === 'calibrated'
              ? `${formatDuration(est.range_s[0])} ~ ${formatDuration(est.range_s[1])} 사이일 가능성이 높습니다.`
              : `아직 완료된 학습이 없어 보정하지 못했습니다 — ${formatDuration(est.range_s[0])} ~ ${formatDuration(est.range_s[1])} 로 크게 벌어질 수 있습니다.`}
          </div>
          {showAssumptions && (
            <ul className="help" style={{ margin: 0, paddingLeft: '1.2em' }}>
              {est.assumptions.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {items.length > 0 && (
        <>
          <table>
            <thead>
              <tr>
                <th>설정</th>
                <th>지금</th>
                <th>제안</th>
                <th>근거</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) =>
                Object.entries(item.changes).map(([key, change], i) => (
                  <tr key={`${item.rule}:${key}`}>
                    <td className="mono">{key}</td>
                    <td className="muted">{show(change.from)}</td>
                    <td className="diff">{show(change.to)}</td>
                    {i === 0 && (
                      <td rowSpan={Object.keys(item.changes).length}>
                        <span className={item.severity === 'warn' ? '' : 'muted'}>{item.reason}</span>
                        <div className="help">{item.effect}</div>
                      </td>
                    )}
                  </tr>
                )),
              )}
            </tbody>
          </table>
          <div className="row tight" style={{ marginTop: 'var(--sp-4)' }}>
            <button onClick={() => onApply(rec?.patch ?? {})}>제안 전체 적용</button>
            <span className="help">적용해도 시작 전까지는 언제든 다시 고칠 수 있습니다.</span>
          </div>
        </>
      )}

      {advisories.map((advisory) => (
        <div key={advisory.code} className={`help ${advisory.severity}`} style={{ marginTop: 'var(--sp-3)' }}>
          {advisory.message}
        </div>
      ))}
    </div>
  )
}
