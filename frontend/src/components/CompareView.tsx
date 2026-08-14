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
import type { Run, TrainEvent } from '../types'

const COLORS = ['#4f8cff', '#35c46b', '#e2b23c', '#b07cff', '#e2564a', '#3fc7c7']
const AXIS = { stroke: '#5b6273', fontSize: 11 }
const TOOLTIP_STYLE = {
  background: '#171a21',
  border: '1px solid #2a2f3a',
  borderRadius: 6,
  fontSize: 12,
}
const METRICS = ['mAP50-95', 'mAP50', 'precision', 'recall'] as const

interface Loaded {
  run: Run
  epochs: TrainEvent[]
}

/**
 * 완료된 run 여러 개를 겹쳐 본다.
 *
 * 전용 비교 API 를 만들지 않는다 — 필요한 건 이미 /api/runs/{id} 와 /api/runs/{id}/events 에 다 있고,
 * 비교 대상은 보통 2~3개라 병렬로 불러도 비용이 없다.
 */
export function CompareView({ runIds }: { runIds: string[] }) {
  const [loaded, setLoaded] = useState<Loaded[]>([])
  const [metric, setMetric] = useState<(typeof METRICS)[number]>('mAP50-95')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!runIds.length) {
      setLoaded([])
      return
    }
    let cancelled = false
    setLoading(true)
    Promise.all(
      runIds.map(async (id) => {
        const [run, events] = await Promise.all([api.run(id), api.events(id)])
        return { run, epochs: events.events.filter((e) => e.t === 'epoch') }
      }),
    )
      .then((rows) => !cancelled && setLoaded(rows))
      .catch(() => !cancelled && setLoaded([]))
      .finally(() => !cancelled && setLoading(false))
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

  if (!runIds.length) return null
  if (loading && !loaded.length) return <div className="card muted">불러오는 중…</div>

  return (
    <>
      <div className="card">
        <h3 className="row" style={{ gap: 8 }}>
          실행 비교 ({loaded.length}개)
          <select
            style={{ width: 140, marginLeft: 'auto' }}
            value={metric}
            onChange={(e) => setMetric(e.target.value as (typeof METRICS)[number])}
          >
            {METRICS.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="#232833" />
            <XAxis dataKey="epoch" {...AXIS} />
            <YAxis domain={[0, 1]} {...AXIS} />
            <Tooltip contentStyle={TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {loaded.map((l, i) => (
              <Line
                key={l.run.id}
                type="monotone"
                dataKey={l.run.id}
                name={l.run.name}
                stroke={COLORS[i % COLORS.length]}
                dot={false}
                strokeWidth={1.8}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>

        <table style={{ marginTop: 8 }}>
          <thead>
            <tr><th>실행</th><th>최고 mAP50-95</th><th>에폭</th></tr>
          </thead>
          <tbody>
            {best.map((b, i) => (
              <tr key={b.id}>
                <td><span style={{ color: COLORS[i % COLORS.length] }}>■</span> {b.name}</td>
                <td>{b.value < 0 ? '-' : b.value.toFixed(4)}</td>
                <td className="muted">{b.epoch || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>다른 설정만</h3>
        {loaded.length < 2 ? (
          <p className="muted small">두 개 이상 선택하면 차이를 비교합니다.</p>
        ) : diff.length === 0 ? (
          <p className="muted small">설정이 완전히 동일합니다.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>파라미터</th>
                {loaded.map((l, i) => (
                  <th key={l.run.id} style={{ color: COLORS[i % COLORS.length] }}>{l.run.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {diff.map((d) => (
                <tr key={d.key}>
                  <td className="mono">{d.key}</td>
                  {d.values.map((v, i) => (
                    <td key={i} className="mono">{v.length > 40 ? `…${v.slice(-38)}` : v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}
