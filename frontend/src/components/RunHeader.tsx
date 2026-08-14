import { useMemo, type ReactNode } from 'react'
import { formatDuration, statusLabel } from '../format'
import type { Run } from '../types'
import type { StreamState } from '../useRunStream'
import { useConfirm } from './ui/Dialog'

interface Props {
  run: Run
  stream: StreamState
  onStop: (mode: 'graceful' | 'force') => void
}

/**
 * 지금 보고 있는 run 의 상태 줄.
 *
 * 예전에는 앱 헤더에 셀렉트·배지·진행률·정지 버튼·GPU 가 한 줄로 뭉쳐 있었다.
 * run 에 딸린 것은 여기로 내리고, 헤더에는 앱 전체에 해당하는 것만 남긴다.
 */
export function RunHeader({ run, stream, onStop }: Props) {
  const confirm = useConfirm()

  const progress = useMemo(() => {
    const start = stream.events.find((e) => e.t === 'start')
    const last = [...stream.events].reverse().find((e) => e.t === 'epoch')
    const total = last?.total_epochs ?? start?.total_epochs ?? 0
    const done = last?.epoch ?? 0
    const batch = stream.batch
    const withinEpoch = batch?.n ? (batch.i ?? 0) / batch.n : 0
    const fraction = total ? Math.min((done + withinEpoch) / total, 1) : 0
    return { total, done, fraction, eta: last?.eta_s ?? null }
  }, [stream.events, stream.batch])

  const stats = useMemo(() => {
    const epochs = stream.events.filter((e) => e.t === 'epoch')
    const last = epochs[epochs.length - 1]

    let bestValue: number | null = null
    let bestEpoch: number | null = null
    for (const e of epochs) {
      const v = e.summary?.['mAP50-95']
      if (v != null && (bestValue == null || v > bestValue)) {
        bestValue = v
        bestEpoch = e.epoch ?? null
      }
    }

    // ultralytics 는 train/box_loss 처럼 나눠 준다. 합쳐야 "지금 손실" 이 된다.
    const trainLossKeys = Object.keys(last?.metrics ?? {}).filter((k) => k.includes('loss') && k.includes('train'))
    const loss = trainLossKeys.length
      ? trainLossKeys.reduce((sum, k) => sum + (last?.metrics?.[k] ?? 0), 0)
      : (stream.batch?.loss ?? null)

    // 시계를 따로 돌리지 않는다 — 학습 중에는 이벤트가 계속 들어와 마지막 시각이 곧 현재다.
    const lastTs = stream.batch?.ts ?? stream.events[stream.events.length - 1]?.ts ?? null
    const endTs = run.finished_at ?? lastTs
    const elapsed = run.started_at && endTs ? endTs - run.started_at : null

    return { bestValue, bestEpoch, loss, elapsed }
  }, [stream.events, stream.batch, run.started_at, run.finished_at])

  const percent = Math.round(progress.fraction * 100)

  async function forceStop() {
    const ok = await confirm({
      title: '강제 종료할까요?',
      body: '프로세스를 즉시 죽입니다. 안전 정지와 달리 best.pt 와 플롯이 남는다는 보장이 없습니다. 에폭이 끝날 때까지 기다릴 수 있다면 안전 정지를 쓰세요.',
      confirmLabel: '강제 종료',
      danger: true,
    })
    if (ok) onStop('force')
  }

  return (
    <header className="run-header">
      <div className="row wrap">
        <strong>{run.name}</strong>
        <span className={`badge ${run.status}`}>{statusLabel(run.status)}</span>

        {progress.total > 0 && (
          <>
            <span className="small muted nowrap">
              {progress.done}/{progress.total} 에폭
            </span>
            <div
              className="progress"
              style={{ maxWidth: 260 }}
              role="progressbar"
              aria-label="학습 진행률"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={percent}
              aria-valuetext={`${progress.done}/${progress.total} 에폭, ${percent}%`}
            >
              <div style={{ width: `${percent}%` }} />
            </div>
            {progress.eta != null && run.status === 'running' && (
              <span className="small muted nowrap">남은 시간 {formatDuration(progress.eta)}</span>
            )}
          </>
        )}

        <span className="row tight spacer">
          {run.status === 'running' && (
            <>
              <button className="btn-sm" onClick={() => onStop('graceful')}>
                안전 정지
              </button>
              <button className="btn-sm danger" onClick={forceStop}>
                강제 종료
              </button>
            </>
          )}
          {run.status === 'queued' && (
            <button className="btn-sm" onClick={() => onStop('graceful')}>
              대기 취소
            </button>
          )}
          <StreamIndicator stream={stream} />
        </span>
      </div>

      <dl className="stat-row">
        <Stat label="에폭" value={progress.total ? `${progress.done}/${progress.total}` : '-'} />
        <Stat
          label="최고 mAP50-95"
          value={stats.bestValue == null ? '-' : stats.bestValue.toFixed(4)}
          unit={stats.bestEpoch != null ? `${stats.bestEpoch}에폭` : undefined}
        />
        <Stat label="최근 손실" value={stats.loss == null ? '-' : stats.loss.toFixed(4)} />
        <Stat label="경과 시간" value={stats.elapsed == null ? '-' : formatDuration(stats.elapsed)} />
      </dl>

      {run.error && <div className="error small">{run.error}</div>}
    </header>
  )
}

function Stat({ label, value, unit }: { label: string; value: ReactNode; unit?: string }) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd>
        {value}
        {unit && <span className="unit">{unit}</span>}
      </dd>
    </div>
  )
}

function StreamIndicator({ stream }: { stream: StreamState }) {
  if (stream.status === 'open') {
    return (
      <span className="small muted nowrap">
        <span className="dot running" aria-hidden="true" /> 연결됨
      </span>
    )
  }
  if (stream.status === 'connecting') {
    return <span className="small muted nowrap">연결 중…</span>
  }
  if (stream.status === 'gone') {
    return <span className="small error nowrap">서버에 없는 실행입니다</span>
  }
  // 끊긴 상태. 자동 재시도가 돌고 있어도 사람이 먼저 누를 수 있게 버튼을 둔다.
  return (
    <span className="row tight nowrap">
      <span className="small muted">
        {stream.finished ? '스트림 종료' : stream.attempt > 0 ? `연결 끊김 · 재시도 ${stream.attempt}회` : '연결 끊김'}
      </span>
      <button className="btn-xs" onClick={stream.reconnect}>
        다시 연결
      </button>
    </span>
  )
}
