import { useEffect, useState } from 'react'

import { api } from '../api'
import type { Dataset, JobStatus, TuneEstimate, TuneReport } from '../types'
import { Field } from './ui/Field'

const POLL_MS = 2000

function duration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}초`
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  if (!hours) return `${minutes}분`
  return `${hours}시간 ${minutes}분`
}

/**
 * 하이퍼파라미터 자동 탐색 — ultralytics `model.tune()`.
 *
 * 다른 사이드잡과 폴링 방식이 하나 다르다. 잡 이벤트에는 "몇 번째 시도인가" 가 없으므로
 * (시도는 별도 프로세스로 돌고 자기 이벤트를 따로 쓴다) **실행 중에도 리포트를 함께 읽는다.**
 * 그래서 리포트가 아직 없는 초반의 404 는 오류가 아니라 정상이다.
 */
export function TunePanel({
  dataset,
  model,
  gpuCount,
  onApply,
}: {
  dataset: Dataset | null | undefined
  model: string
  gpuCount: number
  onApply: (patch: Record<string, unknown>) => void
}) {
  const [job, setJob] = useState<JobStatus | null>(null)
  const [report, setReport] = useState<TuneReport | null>(null)
  const [prediction, setPrediction] = useState<TuneEstimate | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const [iterations, setIterations] = useState(20)
  const [epochs, setEpochs] = useState(10)
  const [fraction, setFraction] = useState(1)
  // null = 아직 사용자가 고르지 않음. useState(gpuCount) 로 두면 **GPU 목록이 도착하기 전
  // 마운트된 값(0)이 그대로 굳어**, 손대지 않고 시작한 몇 시간짜리 탐색이 CPU 로 돈다.
  const [chosenGpus, setGpus] = useState<number | null>(null)
  const [restart, setRestart] = useState(false)

  const gpus = chosenGpus ?? Math.min(1, gpuCount)
  const id = dataset?.id
  const args = { model, iterations, epochs, fraction, gpus, restart }

  useEffect(() => {
    if (!id) return
    let cancelled = false
    setJob(null)
    setReport(null)
    setError('')
    // 데이터셋을 바꾸면 반드시 끈다. 체크박스는 결과가 있을 때만 보이므로, 켜 둔 채 다른
    // 데이터셋으로 넘어가면 보이지도 않는 상태로 그쪽 기록을 지우게 된다.
    setRestart(false)
    api.tuneStatus(id).then((s) => !cancelled && setJob(s)).catch(() => {})
    api.tuneReport(id).then((r) => !cancelled && setReport(r)).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [id])

  useEffect(() => {
    if (!id || job?.status !== 'running') return
    let cancelled = false
    const timer = setInterval(async () => {
      try {
        const next = await api.tuneStatus(id)
        if (cancelled) return
        setJob(next)
        // 진행률이 리포트 안에 있어 실행 중에도 매 주기 읽는다.
        try {
          const fresh = await api.tuneReport(id)
          if (!cancelled) setReport(fresh)
        } catch {
          /* 첫 시도가 끝나기 전에는 리포트가 없다 */
        }
      } catch {
        /* 폴링 실패는 다음 주기에 다시 시도한다 */
      }
    }, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [job?.status, id])

  // 잡이 끝나면 리포트를 한 번 더 읽는다. 거절된 이어하기(설정 불일치)는 서버가 이전 리포트를
  // 지우지 않으므로, 다시 읽지 않으면 화면에서만 결과가 사라진다.
  useEffect(() => {
    if (!id || !job || job.status === 'running' || job.status === 'idle') return
    let cancelled = false
    api.tuneReport(id).then((r) => !cancelled && setReport(r)).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [job?.status, id])

  // 예상 소요는 서버가 계산하고 가정 문장까지 함께 내려준다. 여기서 곱하지 않는다.
  useEffect(() => {
    if (!id || !model) return
    let cancelled = false
    const timer = setTimeout(() => {
      api
        .tuneEstimate(id, { model, iterations, epochs, fraction, gpus })
        .then((r) => !cancelled && setPrediction(r))
        .catch(() => !cancelled && setPrediction(null))
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [id, model, iterations, epochs, fraction, gpus])

  if (!dataset) return null

  const running = job?.status === 'running'

  const start = async () => {
    setBusy(true)
    setError('')
    try {
      const next = await api.startTune(dataset.id, args)
      setJob(next)
      setReport(null)
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    setBusy(true)
    try {
      setJob(await api.stopTune(dataset.id))
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  const done = report?.iterations_done ?? 0
  const target = report?.iterations_target ?? iterations
  const current = report?.current

  return (
    <div className="card">
      <div className="card-head">
        <h3>하이퍼파라미터 탐색</h3>
        <button
          className="btn-sm spacer"
          onClick={running ? stop : start}
          disabled={busy || !model}
        >
          {running ? '정지' : done ? '이어서 탐색' : '탐색 시작'}
        </button>
      </div>
      <p className="help">
        학습률·손실 가중치 같은 값을 바꿔 가며 짧은 학습을 여러 번 돌려 보고, 가장 좋았던 조합을
        찾아 아래 파라미터 폼에 넣어 줍니다. 첫 시도는 지금 기본값 그대로 돌려 비교 기준으로
        씁니다. <strong>탐색이 도는 동안 그 GPU 로는 학습이 시작되지 않습니다.</strong>
      </p>

      {/* 입력 넷은 한 줄에 둔다 — 서로를 보면서 정하는 값이라 2열로 접으면 짝이 어긋난다. */}
      <div className="tune-inputs">
        <Field label="시도 횟수" help="많을수록 좋은 조합을 찾을 확률이 오르고 그만큼 오래 걸립니다.">
          {(props) => (
            <input
              {...props}
              type="number"
              min={2}
              max={200}
              value={iterations}
              disabled={running}
              onChange={(e) => setIterations(Number(e.target.value))}
            />
          )}
        </Field>
        <Field label="시도당 에폭" help="짧게 돌려 순위만 매깁니다. 본 학습의 에폭과는 다릅니다.">
          {(props) => (
            <input
              {...props}
              type="number"
              min={1}
              max={1000}
              value={epochs}
              disabled={running}
              onChange={(e) => setEpochs(Number(e.target.value))}
            />
          )}
        </Field>
        <Field label="데이터 사용 비율" help="일부만 써서 탐색을 줄입니다. 1 이면 전부 씁니다.">
          {(props) => (
            <input
              {...props}
              type="number"
              min={0.01}
              max={1}
              step={0.05}
              value={fraction}
              disabled={running}
              onChange={(e) => setFraction(Number(e.target.value))}
            />
          )}
        </Field>
        <Field label="GPU 장수" help={`이 기계에 ${gpuCount}장 있습니다. 0 이면 CPU 로 돕니다.`}>
          {(props) => (
            <input
              {...props}
              type="number"
              min={0}
              max={Math.max(gpuCount, 1)}
              value={gpus}
              disabled={running}
              onChange={(e) => setGpus(Number(e.target.value))}
            />
          )}
        </Field>
      </div>
      {done > 0 && !running && (
        <label className="small">
          <input
            type="checkbox"
            checked={restart}
            onChange={(e) => setRestart(e.target.checked)}
          />{' '}
          처음부터 다시 (지금까지의 시도 {done}개를 버립니다)
        </label>
      )}

      {prediction && !running && (
        <>
          <p className="small muted">
            {prediction.ok
              ? `예상 소요 약 ${duration(prediction.total_time_s ?? 0)} · ${prediction.note ?? ''}`
              : prediction.reason}
          </p>
          {/* 짧게 잡으면 무엇을 잃는지는 시작 전에 봐야 한다. 몇 시간을 쓰고 나서 알면 늦다. */}
          {prediction.warnings?.map((line) => (
            <p key={line} className="help warn small">
              {line}
            </p>
          ))}
          {prediction.ok && prediction.assumptions && (
            <ul className="small muted">
              {prediction.assumptions.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          )}
        </>
      )}

      {error && <p className="error small">{error}</p>}
      {job?.status === 'failed' && (
        <p className="error small">
          탐색이 실패했습니다: {String(job.result?.error ?? job.error ?? '')}
        </p>
      )}

      {running && (
        <p className="small muted">
          시도 {done + 1}/{target} 진행 중
          {current ? ` · 에폭 ${current.epoch}/${current.total_epochs}` : ' · 준비 중…'}
          {report?.best ? ` · 최고 fitness ${report.best.fitness.toFixed(4)}` : ''}
          {/* 시작 전 추정이 아니라 이번 잡이 실제로 쓴 시간에서 낸 값이다. */}
          {report?.eta_s ? ` · 남은 시간 약 ${duration(report.eta_s)}` : ''}
        </p>
      )}

      {report && done > 0 && (
        <>
          <p className="small muted">
            시도 {done}/{target} 완료
            {report.baseline ? ` · 기본값 ${report.baseline.fitness.toFixed(4)}` : ''}
            {report.best ? ` · 최고 ${report.best.fitness.toFixed(4)} (${report.best.i}번째)` : ''}
            {report.gain !== null && report.gain !== undefined
              ? ` · 개선폭 ${report.gain >= 0 ? '+' : ''}${report.gain.toFixed(4)}`
              : ''}
            {/* 값은 mAP50-95 다. 8.4.47 의 검출 fitness 가중치가 [0,0,0,1] 이라 같은 값이다. */}
            {report.noise
              ? ` · 시드 흔들림 ±${report.noise.stdev.toFixed(4)} · 문턱 ${report.threshold.toFixed(4)}`
              : ''}
          </p>

          {report.advisories.map((line) => (
            <p key={line} className="small muted">
              {line}
            </p>
          ))}

          {report.items.map((item) => (
            <div key={item.rule}>
              <p className="small">{item.reason}</p>
              <p className="small muted">{item.effect}</p>
              <ul className="small">
                {Object.entries(item.changes).map(([key, change]) => (
                  <li key={key}>
                    {key}: {String(change.from)} → <strong>{String(change.to)}</strong>
                  </li>
                ))}
              </ul>
            </div>
          ))}

          {Object.keys(report.patch).length > 0 && (
            <button className="btn-sm" onClick={() => onApply(report.patch)}>
              이 값으로 폼 채우기
            </button>
          )}

          <details>
            <summary className="small">시도별 결과 {report.trials.length}건</summary>
            <table className="small">
              <thead>
                <tr>
                  <th>시도</th>
                  <th>fitness</th>
                  <th>lr0</th>
                  <th>box</th>
                  <th>cls</th>
                </tr>
              </thead>
              <tbody>
                {report.trials.map((trial) => (
                  <tr key={trial.i}>
                    <td>{trial.i === 1 ? '1 (기본값)' : trial.i}</td>
                    <td>{trial.ok ? trial.fitness.toFixed(4) : '실패'}</td>
                    <td>{trial.hyp.lr0?.toFixed(5)}</td>
                    <td>{trial.hyp.box?.toFixed(2)}</td>
                    <td>{trial.hyp.cls?.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </>
      )}
    </div>
  )
}
