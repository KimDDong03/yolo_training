import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { chartAxis, chartGrid, chartLegend, chartTooltip, metricSeries, seriesColor } from '../theme'
import type { TrainEvent } from '../types'

const METRIC_KEYS = ['mAP50', 'mAP50-95', 'precision', 'recall']

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
        ...Object.fromEntries(METRIC_KEYS.map((key) => [key, e.summary?.[key] ?? null])),
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
      <div className="card-head">
        <h3>정확도 지표</h3>
        {last && (
          <span className="muted small spacer" style={{ fontWeight: 400 }}>
            최근 mAP50 {fmt(last['mAP50'] as number | null)} · mAP50-95 {fmt(last['mAP50-95'] as number | null)}
            {best.epoch != null && ` · 최고 ${fmt(best.value)} (${best.epoch}에폭)`}
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid stroke={chartGrid.stroke} />
          <XAxis dataKey="epoch" stroke={chartAxis.stroke} tick={chartAxis.tick} tickLine={chartAxis.tickLine} />
          <YAxis domain={[0, 1]} stroke={chartAxis.stroke} tick={chartAxis.tick} tickLine={chartAxis.tickLine} />
          <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} />
          <Legend wrapperStyle={chartLegend.wrapperStyle} />
          {METRIC_KEYS.map((key) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={metricSeries[key]}
              dot={false}
              strokeWidth={1.8}
              isAnimationActive={false}
              connectNulls
            />
          ))}
          {/* 최고 지점을 찍어 두면 "지금이 최고인가"를 눈으로 바로 판단할 수 있다. */}
          {best.epoch != null && (
            <ReferenceDot
              x={best.epoch}
              y={best.value}
              r={4}
              fill={metricSeries['mAP50-95']}
              stroke="#212328"
              strokeWidth={2}
              isFront
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export function LossChart({ events }: Props) {
  // 손실은 초반에 급락해 뒷부분이 눌린다. 로그 축이 후반 수렴을 보기에 낫다.
  const [logScale, setLogScale] = useState(false)

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

  return (
    <div className="card">
      <div className="card-head">
        <h3>손실</h3>
        <span className="segmented spacer" role="radiogroup" aria-label="손실 축 눈금">
          {([false, true] as const).map((v) => (
            <button
              key={String(v)}
              type="button"
              role="radio"
              aria-checked={logScale === v}
              onClick={() => setLogScale(v)}
            >
              {v ? '로그' : '선형'}
            </button>
          ))}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid stroke={chartGrid.stroke} />
          <XAxis dataKey="epoch" stroke={chartAxis.stroke} tick={chartAxis.tick} tickLine={chartAxis.tickLine} />
          <YAxis
            scale={logScale ? 'log' : 'linear'}
            domain={logScale ? ['auto', 'auto'] : [0, 'auto']}
            allowDataOverflow={false}
            stroke={chartAxis.stroke}
            tick={chartAxis.tick}
            tickLine={chartAxis.tickLine}
          />
          <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} />
          <Legend wrapperStyle={chartLegend.wrapperStyle} />
          {series.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={seriesColor(i)}
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
