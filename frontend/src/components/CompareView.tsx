import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api'
import { chartAxis, chartGrid, chartLegend, chartTooltip, seriesColor } from '../theme'
import type { Run, TrainEvent } from '../types'
import { EmptyState, SkeletonRows } from './ui/EmptyState'

const METRICS = ['mAP50-95', 'mAP50', 'precision', 'recall'] as const

interface Loaded {
  run: Run
  epochs: TrainEvent[]
}

/**
 * 완료된 run 여러 개를 겹쳐 본다.
 *
 * 전용 비교 API 를 만들지 않는다 — 필요한 건 이미 /api/runs/{id} 와 /api/runs/{id}/events 에 다 있다.
 * 선택 개수는 사이드바에서 COMPARE_LIMIT 로 묶여 있어 병렬 요청이 무한히 늘지 않는다.
 */
export function CompareView({ runIds }: { runIds: string[] }) {
  const [loaded, setLoaded] = useState<Loaded[]>([])
  const [failed, setFailed] = useState<string[]>([])
  const [metric, setMetric] = useState<(typeof METRICS)[number]>('mAP50-95')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!runIds.length) {
      setLoaded([])
      setFailed([])
      return
    }
    let cancelled = false
    setLoading(true)
    // allSettled 다 — 하나가 실패해도 나머지는 그린다. 예전에는 삭제된 run 하나가
    // 선택에 남아 있으면 비교 화면 전체가 빈 채로 떴다.
    Promise.allSettled(
      runIds.map(async (id) => {
        const [run, events] = await Promise.all([api.run(id), api.events(id)])
        return { run, epochs: events.events.filter((e) => e.t === 'epoch') }
      }),
    ).then((results) => {
      if (cancelled) return
      setLoaded(results.flatMap((r) => (r.status === 'fulfilled' ? [r.value] : [])))
      setFailed(runIds.filter((_, i) => results[i].status === 'rejected'))
      setLoading(false)
    })
    return () => {
      cancelled = true
    }
  }, [runIds.join(',')])

  const data = useMemo(() => {
    const maxEpoch = Math.max(0, ...loaded.map((l) => l.epochs.length))
    const rows: Record<string, number | null>[] = []
    for (let i = 0; i < maxEpoch; i++) {
      const row: Record<string, number | null> = { epoch: i + 1 }
      for (const l of loaded) {
        row[l.run.id] = (l.epochs[i]?.summary?.[metric] as number | undefined) ?? null
      }
      rows.push(row)
    }
    return rows
  }, [loaded, metric])

  // 값이 다른 파라미터만 추린다. 45개를 다 나열하면 아무도 안 본다.
  const diff = useMemo(() => {
    if (loaded.length < 2) return []
    const keys = new Set<string>()
    for (const l of loaded) {
      Object.keys(l.run.params).forEach((k) => keys.add(k))
      Object.keys(l.run.options ?? {}).forEach((k) => keys.add(k))
    }
    const rows: { key: string; values: string[] }[] = []
    for (const key of [...keys].sort()) {
      const values = loaded.map((l) => {
        const v = (l.run.params as Record<string, unknown>)[key] ?? (l.run.options ?? {})[key]
        return v === undefined || v === null ? '-' : String(v)
      })
      if (new Set(values).size > 1) rows.push({ key, values })
    }
    return rows
  }, [loaded])

  const best = useMemo(
    () =>
      loaded.map((l) => {
        let value = -1
        let epoch = 0
        l.epochs.forEach((e, i) => {
          const v = e.summary?.['mAP50-95']
          if (typeof v === 'number' && v > value) {
            value = v
            epoch = i + 1
          }
        })
        return { id: l.run.id, name: l.run.name, value, epoch }
      }),
    [loaded],
  )

  if (!runIds.length) {
    return <EmptyState title="비교할 실행을 고르세요" description="왼쪽 목록에서 체크박스로 두 개 이상 선택합니다." />
  }

  if (loading && !loaded.length) {
    return (
      <div className="card">
        <h3>실행 비교</h3>
        <SkeletonRows rows={4} />
      </div>
    )
  }

  return (
    <>
      {failed.length > 0 && (
        <div className="card error small">
          {failed.length}개 실행을 불러오지 못했습니다 (지워졌을 수 있습니다): <span className="mono">{failed.join(', ')}</span>
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <h3>실행 비교 ({loaded.length}개)</h3>
          <div className="segmented spacer" role="radiogroup" aria-label="비교할 지표">
            {METRICS.map((m) => (
              <button
                key={m}
                type="button"
                role="radio"
                aria-checked={metric === m}
                className="tiny mono"
                onClick={() => setMetric(m)}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid stroke={chartGrid.stroke} />
            <XAxis dataKey="epoch" stroke={chartAxis.stroke} tick={chartAxis.tick} tickLine={chartAxis.tickLine} />
            <YAxis domain={[0, 1]} stroke={chartAxis.stroke} tick={chartAxis.tick} tickLine={chartAxis.tickLine} />
            <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} />
            <Legend wrapperStyle={chartLegend.wrapperStyle} />
            {loaded.map((l) => (
              <Line
                key={l.run.id}
                type="monotone"
                dataKey={l.run.id}
                name={l.run.name}
                stroke={seriesColor(runIds.indexOf(l.run.id))}
                dot={false}
                strokeWidth={2.2}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>

      </div>

      {/* 결론(어느 실행이 이겼나)과 이유(무엇이 달랐나)를 나란히 둔다. */}
      <div className="compare-grid">
        <div className="card">
          <h3>최고 성능</h3>
          <ul className="best-list">
            {best.map((b) => (
              <li key={b.id}>
                <span className="best-bar" style={{ background: seriesColor(runIds.indexOf(b.id)) }} aria-hidden="true" />
                <span className="best-name">{b.name}</span>
                <span className="best-value">{b.value < 0 ? '-' : b.value.toFixed(4)}</span>
                <span className="best-epoch">{b.epoch ? `${b.epoch}에폭` : '-'}</span>
              </li>
            ))}
          </ul>
        </div>

      <div className="card">
        <h3>다른 설정만</h3>
        {loaded.length < 2 ? (
          <p className="muted small">두 개 이상 선택하면 차이를 비교합니다.</p>
        ) : diff.length === 0 ? (
          <p className="muted small">설정이 완전히 동일합니다.</p>
        ) : (
          <table>
            <caption className="sr-only">값이 서로 다른 파라미터. 첫 실행과 다른 값은 강조 표시된다.</caption>
            <thead>
              <tr>
                <th scope="col">파라미터</th>
                {loaded.map((l) => (
                  <th key={l.run.id} scope="col" style={{ color: seriesColor(runIds.indexOf(l.run.id)) }}>
                    {l.run.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {diff.map((d) => (
                <tr key={d.key}>
                  <td className="mono">{d.key}</td>
                  {d.values.map((v, i) => (
                    // 기준(첫 실행)과 다른 칸만 강조한다. 다 칠하면 어디가 기준인지 사라진다.
                    <td key={i} className={`mono ${i > 0 && v !== d.values[0] ? 'diff' : ''}`}>
                      {v.length > 40 ? `…${v.slice(-38)}` : v}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        </div>
      </div>
    </>
  )
}
