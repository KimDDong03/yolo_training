import { useMemo } from 'react'
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
import type { TrainEvent } from '../types'

const METRIC_SERIES = [
  { key: 'mAP50', color: '#4f8cff' },
  { key: 'mAP50-95', color: '#35c46b' },
  { key: 'precision', color: '#e2b23c' },
  { key: 'recall', color: '#b07cff' },
]

const AXIS = { stroke: '#5b6273', fontSize: 11 }
const TOOLTIP_STYLE = {
  background: '#171a21',
  border: '1px solid #2a2f3a',
  borderRadius: 6,
  fontSize: 12,
}

interface Props {
  events: TrainEvent[]
}

function epochRows(events: TrainEvent[]) {
  return events.filter((e) => e.t === 'epoch')
}

type MetricRow = { epoch: number | undefined } & Record<string, number | null | undefined>

export function MetricsChart({ events }: Props) {
  const data = useMemo<MetricRow[]>(
    () =>
      epochRows(events).map((e) => ({
        epoch: e.epoch,
        ...Object.fromEntries(METRIC_SERIES.map((s) => [s.key, e.summary?.[s.key] ?? null])),
      })),
    [events],
  )

  const best = useMemo(() => {
    let bestEpoch: number | null = null
    let bestValue = -1
    for (const row of data) {
      const v = row['mAP50-95'] as number | null
      if (v != null && v > bestValue) {
        bestValue = v
        bestEpoch = row.epoch as number
      }
    }
    return { epoch: bestEpoch, value: bestValue }
  }, [data])

  const last = data[data.length - 1]

  return (
    <div className="card">
      <h3>
        정확도 지표
        {last && (
          <span className="muted" style={{ float: 'right', fontWeight: 400 }}>
            최근 mAP50 {fmt(last['mAP50'] as number | null)} · mAP50-95 {fmt(last['mAP50-95'] as number | null)}
            {best.epoch != null && ` · 최고 ${fmt(best.value)} (${best.epoch}에폭)`}
          </span>
        )}
      </h3>
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid stroke="#232833" />
          <XAxis dataKey="epoch" {...AXIS} />
          <YAxis domain={[0, 1]} {...AXIS} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {METRIC_SERIES.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={s.color}
              dot={false}
              strokeWidth={1.8}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export function LossChart({ events }: Props) {
  const { data, series } = useMemo(() => {
    const rows = epochRows(events)
    const keys = new Set<string>()
    rows.forEach((e) => Object.keys(e.metrics ?? {}).forEach((k) => k.includes('loss') && keys.add(k)))
    const ordered = [...keys].sort()
    return {
      data: rows.map((e) => ({
        epoch: e.epoch,
        ...Object.fromEntries(ordered.map((k) => [k, e.metrics?.[k] ?? null])),
      })),
      series: ordered,
    }
  }, [events])

  const colors = ['#4f8cff', '#35c46b', '#e2b23c', '#b07cff', '#e2564a', '#3fc7c7']

  return (
    <div className="card">
      <h3>손실</h3>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid stroke="#232833" />
          <XAxis dataKey="epoch" {...AXIS} />
          <YAxis {...AXIS} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {series.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={colors[i % colors.length]}
              dot={false}
              strokeWidth={1.6}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function fmt(v: number | null | undefined) {
  return v == null ? '-' : v.toFixed(4)
}
