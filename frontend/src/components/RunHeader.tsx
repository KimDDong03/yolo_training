import { useMemo } from 'react'
import { statusLabel } from '../format'
import type { Dataset, Run, TrainEvent } from '../types'
import type { StreamState } from '../useRunStream'
import { useConfirm } from './ui/Dialog'

interface Props {
  run: Run
  /** 목록 응답에는 데이터셋이 없다. 상세를 받아 오는 App 이 따로 내려준다. */
  dataset: Dataset | null | undefined
  stream: StreamState
  onStop: (mode: 'graceful' | 'force') => void
}

/** `18:42` — 남은 시간은 폭이 좁고 자주 바뀌어서 "18분 42초" 보다 시계 표기가 읽기 쉽다. */
function clock(seconds: number) {
  const s = Math.max(0, Math.round(seconds))
  const parts = [Math.floor(s / 3600), Math.floor((s % 3600) / 60), s % 60]
  const shown = parts[0] > 0 ? parts : parts.slice(1)
  return shown.map((n, i) => (i === 0 ? String(n) : String(n).padStart(2, '0'))).join(':')
}

function wallClock(inSeconds: number) {
  return new Date(Date.now() + inSeconds * 1000).toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

/** ultralytics 는 train/box_loss 처럼 나눠 준다. 합쳐야 "지금 손실" 이 된다. */
function trainLoss(event: TrainEvent | undefined) {
  const keys = Object.keys(event?.metrics ?? {}).filter((k) => k.includes('loss') && k.includes('train'))
  if (!keys.length) return null
  return keys.reduce((sum, k) => sum + (event?.metrics?.[k] ?? 0), 0)
}

/**
 * 지금 보고 있는 run 의 상태 줄.
 *
 * 이 화면이 먼저 답해야 하는 질문은 "지금 잘 되고 있나" 다. 그래서 크기를 진행률과
 * mAP·손실 두 지표에 몰아주고, 나머지(데이터셋·모델·GPU)는 이름 아래 한 줄로 내렸다.
 * 예전의 동등한 stat 4개는 무엇을 먼저 봐야 하는지를 알려주지 않았다.
 */
export function RunHeader({ run, dataset, stream, onStop }: Props) {
  const confirm = useConfirm()

  const progress = useMemo(() => {
    const start = stream.events.find((e) => e.t === 'start')
    const last = [...stream.events].reverse().find((e) => e.t === 'epoch')
    const end = [...stream.events].reverse().find((e) => e.t === 'end')
    const total = last?.total_epochs ?? start?.total_epochs ?? 0
    const done = last?.epoch ?? 0
    const batch = stream.batch
    const withinEpoch = batch?.n ? (batch.i ?? 0) / batch.n : 0

    /*
     * 시간 예산이 걸린 실행은 에폭 수가 실행 중에 다시 계산된다(trainer.py:546).
     * 그 분모로 바를 그리면 예산 진행률이 아니라 48 -> 47 로 흔들리는 값이 된다.
     * 예산이 켜져 있으면 분모를 시간으로 바꾼다.
     *
     * 경과는 start 이벤트 시각부터 잰다. 그 시각은 ultralytics 가 재는 학습 시작보다
     * 조금 이르다 - 프로세스 기동과 데이터 스캔이 앞에 들어간다. 그만큼 바가 살짝
     * 앞서지만, 없는 정확도를 지어내느니 이 근사를 쓰고 밝혀 둔다.
     */
    const budget = Number(run.params?.time ?? 0) * 3600
    if (budget > 0) {
      const elapsed =
        typeof end?.elapsed_s === 'number'
          ? end.elapsed_s
          : typeof start?.ts === 'number'
            ? Date.now() / 1000 - start.ts
            : 0
      return {
        total,
        done,
        budget,
        elapsed,
        fraction: Math.min(Math.max(elapsed, 0) / budget, 1),
        eta: Math.max(budget - elapsed, 0),
      }
    }

    const fraction = total ? Math.min((done + withinEpoch) / total, 1) : 0
    return { total, done, budget: 0, elapsed: 0, fraction, eta: last?.eta_s ?? null }
  }, [stream.events, stream.batch, run.params])

  const kpi = useMemo(() => {
    const epochs = stream.events.filter((e) => e.t === 'epoch')
    const last = epochs[epochs.length - 1]
    const prev = epochs[epochs.length - 2]

    let bestValue: number | null = null
    let bestEpoch: number | null = null
    for (const e of epochs) {
      const v = e.summary?.['mAP50-95']
      if (v != null && (bestValue == null || v > bestValue)) {
        bestValue = v
        bestEpoch = e.epoch ?? null
      }
    }

    const map = last?.summary?.['mAP50-95'] ?? null
    const mapPrev = prev?.summary?.['mAP50-95'] ?? null
    const loss = trainLoss(last)
    const lossPrev = trainLoss(prev)

    return {
      map,
      mapDelta: map != null && mapPrev != null ? map - mapPrev : null,
      bestValue,
      bestEpoch,
      loss,
      lossDelta: loss != null && lossPrev != null ? loss - lossPrev : null,
    }
  }, [stream.events])

  const percent = Math.round(progress.fraction * 100)
  const running = run.status === 'running'

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
        <h2 className="run-title">{run.name}</h2>
        <span className={`badge ${run.status}`}>{statusLabel(run.status)}</span>
        <span className="run-sub">
          {[dataset?.name, modelName(run), `GPU #${run.devices.join(', #')}`].filter(Boolean).join(' · ')}
        </span>

        <span className="row tight spacer">
          <StreamIndicator stream={stream} />
          {running && (
            <>
              <button onClick={() => onStop('graceful')}>안전 정지</button>
              <button className="danger" onClick={forceStop}>
                강제 종료
              </button>
            </>
          )}
          {run.status === 'queued' && <button onClick={() => onStop('graceful')}>대기 취소</button>}
        </span>
      </div>

      <div className="run-progress">
        <div className="track">
          <div className="row" style={{ alignItems: 'baseline' }}>
            {progress.budget > 0 ? (
              <>
                <span className="epoch-now">{clock(progress.elapsed)}</span>
                <span className="epoch-total">/ {clock(progress.budget)} 예산</span>
                <span className="small muted nowrap">{progress.done || '-'}에폭째</span>
              </>
            ) : (
              <>
                <span className="epoch-now">{progress.total ? progress.done : '-'}</span>
                <span className="epoch-total">/ {progress.total || '-'} 에폭</span>
              </>
            )}
            {progress.eta != null && running && (
              <span className="small muted nowrap spacer">
                남은 시간 <span className="mono">{clock(progress.eta)}</span> · 종료 예정{' '}
                <span className="mono">{wallClock(progress.eta)}</span>
              </span>
            )}
          </div>
          <div
            className="progress tall"
            style={{ marginTop: 10 }}
            role="progressbar"
            aria-label="학습 진행률"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent}
            aria-valuetext={
              progress.budget > 0
                ? `시간 예산 ${clock(progress.budget)} 중 ${clock(progress.elapsed)} 경과, ${percent}%`
                : `${progress.done}/${progress.total} 에폭, ${percent}%`
            }
          >
            <div style={{ width: `${percent}%` }} />
          </div>
        </div>

        <dl className="kpi-row">
          <div className="kpi">
            <dt>mAP50-95</dt>
            <dd>
              <span className="kpi-value">{kpi.map == null ? '-' : kpi.map.toFixed(3)}</span>
              <Delta value={kpi.mapDelta} digits={3} higherIsBetter />
              <span className="kpi-sub">
                {kpi.bestValue == null
                  ? '아직 없음'
                  : `최고 ${kpi.bestValue.toFixed(4)}${kpi.bestEpoch != null ? ` · ${kpi.bestEpoch}에폭` : ''}`}
              </span>
            </dd>
          </div>
          <div className="kpi">
            <dt>손실</dt>
            <dd>
              <span className="kpi-value">{kpi.loss == null ? '-' : kpi.loss.toFixed(3)}</span>
              <Delta value={kpi.lossDelta} digits={3} higherIsBetter={false} />
              <span className="kpi-sub">직전 에폭 대비</span>
            </dd>
          </div>
        </dl>
      </div>

      <Warnings stream={stream} />

      {/* 실패한 run 의 오류는 아래 FailureCard 가 원인·처방과 함께 보여준다(원문은 거기 접혀 있다).
          여기서 또 뿌리면 같은 문장이 두 번 나온다. */}
      {run.error && run.status !== 'failed' && <div className="error small">{run.error}</div>}
    </header>
  )
}

/** 모델 경로는 길다. 파일 이름만 남긴다 — 어느 가중치로 돌았는지는 그것으로 안다. */
function modelName(run: Run) {
  const raw = run.params?.model
  if (typeof raw !== 'string') return null
  return raw.split(/[\\/]/).pop() || raw
}

/**
 * 직전 에폭 대비 증감.
 *
 * 색은 방향이 아니라 좋고 나쁨을 뜻한다 — 손실은 내려가야 좋으므로 ▼ 가 초록이다.
 * 방향으로 칠하면 두 지표가 정반대의 뜻을 같은 색으로 말하게 된다.
 */
function Delta({ value, digits, higherIsBetter }: { value: number | null; digits: number; higherIsBetter: boolean }) {
  if (value == null || value === 0) return null
  const up = value > 0
  const good = up === higherIsBetter
  return (
    <span className={`kpi-delta ${good ? 'good' : 'bad'}`}>
      {up ? '▲' : '▼'} {Math.abs(value).toFixed(digits)}
    </span>
  )
}

/**
 * 학습 중에 백엔드가 발견한 이상.
 *
 * 판정도 문장도 서버가 만든다. 여기는 렌더링만 한다 — 기준을 고칠 때 프론트를 다시
 * 빌드해 반입하지 않아도 되고, 사후 진단(FailureCard)과 같은 규칙을 쓰게 된다.
 */
function Warnings({ stream }: { stream: StreamState }) {
  const warnings = stream.events.filter((e) => e.t === 'warning')
  if (!warnings.length) return null
  return (
    <div className="stack" style={{ gap: 'var(--sp-1)' }}>
      {warnings.map((w) => (
        <div key={w.code ?? w.ts} className={`run-warn${w.severity === 'critical' ? ' bad' : ''}`}>
          <span className="dot" aria-hidden="true" />
          <span>
            <strong style={{ fontWeight: 600 }}>{w.message}</strong>
            {w.hint && <span className="muted"> {w.hint}</span>}
          </span>
        </div>
      ))}
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
