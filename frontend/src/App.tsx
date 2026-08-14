import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from './api'
import { LossChart, MetricsChart } from './components/Charts'
import { LogView } from './components/LogView'
import { NewRunPanel } from './components/NewRunPanel'
import { PreviewPanel } from './components/PreviewPanel'
import type { Dataset, Gpu, Run } from './types'
import { useRunStream } from './useRunStream'

export default function App() {
  const [runs, setRuns] = useState<Run[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [gpus, setGpus] = useState<Gpu[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<Run | null>(null)
  const [error, setError] = useState('')

  const refreshRuns = useCallback(() => api.runs().then(setRuns).catch(() => {}), [])
  const refreshDatasets = useCallback(() => api.datasets().then(setDatasets).catch(() => {}), [])

  useEffect(() => {
    refreshRuns()
    refreshDatasets()
  }, [refreshRuns, refreshDatasets])

  // 실행 목록과 GPU 상태는 가볍게 주기 폴링한다. 무거운 스트림은 WebSocket 이 담당한다.
  useEffect(() => {
    const tick = () => {
      refreshRuns()
      api.gpus().then((r) => setGpus(r.gpus)).catch(() => {})
    }
    tick()
    const timer = setInterval(tick, 2000)
    return () => clearInterval(timer)
  }, [refreshRuns])

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      return
    }
    api.run(selected).then(setDetail).catch(() => setDetail(null))
  }, [selected, runs.find((r) => r.id === selected)?.status])

  const stream = useRunStream(selected)
  const current = runs.find((r) => r.id === selected) ?? null

  const progress = useMemo(() => {
    const start = stream.events.find((e) => e.t === 'start')
    const last = [...stream.events].reverse().find((e) => e.t === 'epoch')
    const total = last?.total_epochs ?? start?.total_epochs ?? 0
    const done = last?.epoch ?? 0
    const batch = stream.batch
    const withinEpoch = batch?.n ? (batch.i ?? 0) / batch.n : 0
    const fraction = total ? Math.min((done + withinEpoch) / total, 1) : 0
    return { total, done, fraction, eta: last?.eta_s ?? null, batch }
  }, [stream.events, stream.batch])

  async function stop(mode: 'graceful' | 'force') {
    if (!selected) return
    try {
      await api.stopRun(selected, mode)
      refreshRuns()
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>YOLO 학습 콘솔</h1>

        <select value={selected ?? ''} onChange={(e) => setSelected(e.target.value || null)} style={{ width: 320 }}>
          <option value="">＋ 새 학습 설정</option>
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.name} · {r.id} · {r.status}
            </option>
          ))}
        </select>

        {current && (
          <>
            <span className={`badge ${current.status}`}>{statusLabel(current.status)}</span>
            {progress.total > 0 && (
              <>
                <span className="small muted">
                  {progress.done}/{progress.total} 에폭
                </span>
                <div className="progress" style={{ maxWidth: 220 }}>
                  <div style={{ width: `${progress.fraction * 100}%` }} />
                </div>
                {progress.eta != null && current.status === 'running' && (
                  <span className="small muted">남은 시간 {formatEta(progress.eta)}</span>
                )}
              </>
            )}
            {current.status === 'running' && (
              <>
                <button onClick={() => stop('graceful')}>안전 정지</button>
                <button className="danger" onClick={() => stop('force')}>강제 종료</button>
              </>
            )}
            {current.status === 'queued' && <button onClick={() => stop('graceful')}>대기 취소</button>}
            <span className="small muted">{stream.connected ? '● 연결됨' : '○ 연결 끊김'}</span>
          </>
        )}

        <span className="small muted" style={{ marginLeft: 'auto' }}>
          {gpus.map((g) => `#${g.index} ${g.utilization}% ${Math.round(g.memory_used_mb / 1024)}/${Math.round(g.memory_total_mb / 1024)}GB`).join('  ')}
        </span>
      </header>

      {error && <div className="card error" style={{ margin: 10 }} onClick={() => setError('')}>{error}</div>}

      <div className="body">
        <div className="left">
          {selected ? (
            <PreviewPanel runId={selected} events={stream.events} dataset={detail?.dataset} />
          ) : (
            <NewRunPanel
              datasets={datasets}
              gpus={gpus}
              onDatasetsChanged={refreshDatasets}
              onStarted={(id) => {
                refreshRuns()
                setSelected(id)
              }}
            />
          )}
        </div>

        <div className="right">
          {selected ? (
            <div className="pane" style={{ display: 'flex', flexDirection: 'column' }}>
              <MetricsChart events={stream.events} />
              <LossChart events={stream.events} />
              {current?.error && <div className="card error small">{current.error}</div>}
              <LogView lines={stream.logs} />
            </div>
          ) : (
            <div className="pane">
              <div className="card">
                <h3>실행 기록</h3>
                {runs.length === 0 ? (
                  <p className="muted small">아직 학습 기록이 없습니다. 왼쪽에서 데이터셋을 등록하고 시작하세요.</p>
                ) : (
                  <table>
                    <thead>
                      <tr><th>이름</th><th>상태</th><th>GPU</th><th>시작</th><th></th></tr>
                    </thead>
                    <tbody>
                      {runs.map((r) => (
                        <tr key={r.id}>
                          <td style={{ cursor: 'pointer' }} onClick={() => setSelected(r.id)}>{r.name}</td>
                          <td><span className={`badge ${r.status}`}>{statusLabel(r.status)}</span></td>
                          <td>{r.devices.length ? r.devices.join(',') : 'cpu'}</td>
                          <td className="muted">{new Date(r.created_at * 1000).toLocaleString('ko-KR')}</td>
                          <td>
                            <button
                              style={{ fontSize: 11, padding: '2px 8px' }}
                              disabled={r.status === 'running' || r.status === 'queued'}
                              onClick={async () => {
                                await api.deleteRun(r.id).catch((e) => setError(String(e)))
                                refreshRuns()
                              }}
                            >
                              삭제
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>

              <div className="card">
                <h3>등록된 데이터셋</h3>
                {datasets.length === 0 ? (
                  <p className="muted small">없습니다.</p>
                ) : (
                  <table>
                    <thead>
                      <tr><th>이름</th><th>출처</th><th>이미지</th><th>클래스</th><th></th></tr>
                    </thead>
                    <tbody>
                      {datasets.map((d) => (
                        <tr key={d.id}>
                          <td>{d.name}</td>
                          <td className="muted">{d.source === 'zip' ? 'zip 업로드' : '경로 참조'}</td>
                          <td>{d.report.total_images.toLocaleString()}</td>
                          <td className="muted">{d.classes.join(', ')}</td>
                          <td>
                            <button
                              style={{ fontSize: 11, padding: '2px 8px' }}
                              onClick={async () => {
                                await api.deleteDataset(d.id).catch((e) => setError(String(e instanceof Error ? e.message : e)))
                                refreshDatasets()
                              }}
                            >
                              삭제
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function statusLabel(status: string) {
  return { queued: '대기', running: '학습중', completed: '완료', stopped: '정지됨', failed: '실패' }[status] ?? status
}

function formatEta(seconds: number) {
  const s = Math.max(0, Math.round(seconds))
  const m = Math.floor(s / 60)
  const h = Math.floor(m / 60)
  if (h > 0) return `${h}시간 ${m % 60}분`
  if (m > 0) return `${m}분 ${s % 60}초`
  return `${s}초`
}
