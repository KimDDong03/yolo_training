import { useEffect, useState } from 'react'

import { api } from '../api'
import type { Diagnosis, Run } from '../types'
import { useToast } from './ui/Toast'

/** 값 하나를 사람이 읽는 문자열로. 배열(devices)과 불리언(amp)이 섞여 온다. */
function show(value: unknown): string {
  if (value === null || value === undefined || value === '') return '없음'
  if (typeof value === 'boolean') return value ? '켬' : '끔'
  if (Array.isArray(value)) return value.length ? `GPU ${value.join(', ')}` : 'CPU'
  return String(value)
}

/**
 * 실패한 학습의 원인·처방과 재시도 버튼.
 *
 * 지금까지는 run.error 원문 한 줄이 전부였다. 원문은 접어서 남겨 두되,
 * 처음 눈에 들어오는 것은 "무엇이 잘못됐고 다음에 뭘 하라" 여야 한다.
 */
export function FailureCard({ run, onStarted }: { run: Run; onStarted: (runId: string) => void }) {
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null)
  const [failed, setFailed] = useState(false)
  const [busy, setBusy] = useState(false)
  const toast = useToast()

  useEffect(() => {
    let cancelled = false
    setDiagnosis(null)
    setFailed(false)
    api
      .diagnosis(run.id)
      .then((d) => !cancelled && setDiagnosis(d))
      // 진단은 실패해도 학습 화면 자체는 계속 봐야 한다. 다만 조용히 삼키지는 않는다 —
      // 아래에서 원문 오류라도 보여준다.
      .catch(() => !cancelled && setFailed(true))
    return () => {
      cancelled = true
    }
  }, [run.id])

  const retry = async () => {
    if (!diagnosis?.retry) return
    setBusy(true)
    try {
      const created = await api.createRun({
        dataset_id: run.dataset_id,
        name: `${run.name} (재시도)`,
        devices: diagnosis.retry.devices,
        params: diagnosis.retry.params,
        options: diagnosis.retry.options,
        retry_of: run.id,
      })
      onStarted(created.id)
    } catch (e) {
      toast(String(e instanceof Error ? e.message : e), 'error')
    } finally {
      setBusy(false)
    }
  }

  // 진단을 못 받아왔으면 최소한 원문은 보여준다.
  if (!diagnosis) {
    return failed && run.error ? <div className="card error small">{run.error}</div> : null
  }

  const changes = Object.entries(diagnosis.retry?.changed ?? {})

  return (
    <div className="card">
      {/* 원문 오류가 아니라 진단 문장이 먼저다. 원문은 아래 접힌 블록에 그대로 남는다. */}
      <div className="diag-label">진단</div>
      <h3 className="diag-title">{diagnosis.title}</h3>

      <p className="diag-body">{diagnosis.cause}</p>
      <p className="diag-body muted">{diagnosis.fix}</p>

      {diagnosis.retry && (
        <>
          {changes.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th>바꿀 설정</th>
                  <th>지금</th>
                  <th>재시도</th>
                </tr>
              </thead>
              <tbody>
                {changes.map(([key, change]) => (
                  <tr key={key}>
                    <td className="mono">{key}</td>
                    <td className="muted">{show(change.from)}</td>
                    <td className="diff">{show(change.to)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="row tight" style={{ marginTop: 'var(--sp-4)' }}>
            <button className="primary" onClick={retry} disabled={busy}>
              {busy ? '시작하는 중…' : diagnosis.retry.label}
            </button>
            <span className="help">
              원래 학습 기록은 남기고 새 학습을 만듭니다.
            </span>
          </div>
        </>
      )}

      {(diagnosis.evidence.length > 0 || diagnosis.log_tail.length > 0) && (
        <details style={{ marginTop: 'var(--sp-4)' }}>
          <summary className="small muted">원문 보기</summary>
          {/* 오류 줄만 색을 준다. 전부 빨갛게 하면 어디가 문제인지 다시 못 찾는다. */}
          {diagnosis.evidence.length > 0 && <LogLines lines={diagnosis.evidence} />}
          {diagnosis.log_tail.length > 0 && <LogLines lines={diagnosis.log_tail} />}
        </details>
      )}
    </div>
  )
}

/** 원문 로그. Error/Traceback 같은 줄만 --err 로 남기고 나머지는 그대로 둔다. */
function LogLines({ lines }: { lines: string[] }) {
  return (
    <pre className="log small">
      {lines.map((line, i) => (
        <div key={i} className={/error|traceback|exception|failed/i.test(line) ? 'err' : undefined}>
          {line}
        </div>
      ))}
    </pre>
  )
}
