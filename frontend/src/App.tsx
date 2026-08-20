import { useEffect, useState } from 'react'
import { api } from './api'
import { LossChart, MetricsChart } from './components/Charts'
import { CompareView } from './components/CompareView'
import { DatasetsView } from './components/DatasetsView'
import { LogView } from './components/LogView'
import { NewRunPanel } from './components/NewRunPanel'
import { PreviewPanel } from './components/PreviewPanel'
import { FailureCard } from './components/FailureCard'
import { RunHeader } from './components/RunHeader'
import { COMPARE_LIMIT, Sidebar } from './components/Sidebar'
import { SplitPane } from './components/SplitPane'
import { useConfirm } from './components/ui/Dialog'
import { useToast } from './components/ui/Toast'
import type { Dataset, Gpu, Run, View } from './types'
import { useResource } from './useResource'
import { useRunStream } from './useRunStream'

/** 폴링이 이만큼 연속으로 실패하면 배너를 띄운다. 한두 번 삐끗한 것으로는 소란 떨지 않는다. */
const OFFLINE_AFTER = 3

export default function App() {
  const [view, setView] = useState<View>({ kind: 'new' })
  const [compare, setCompare] = useState<string[]>([])
  const [detail, setDetail] = useState<Run | null>(null)

  const confirm = useConfirm()
  const toast = useToast()

  const runs = useResource<Run[]>(() => api.runs(), [], 2000)
  const gpus = useResource<Gpu[]>(() => api.gpus().then((r) => r.gpus), [], 2000)
  const datasets = useResource<Dataset[]>(() => api.datasets(), [])

  const runId = view.kind === 'run' ? view.id : null
  const current = runId ? (runs.data.find((r) => r.id === runId) ?? null) : null
  const stream = useRunStream(runId)

  /*
   * 사이드바가 항상 보이면서 지금 보고 있는 run 도 바로 지울 수 있게 됐다.
   * 사라진 id 를 계속 가리키면 상세는 404, WebSocket 은 4404 를 받는다. 여기서 정리한다.
   */
  useEffect(() => {
    if (runs.status !== 'ready') return // 로딩 중 빈 목록으로 화면을 튕기면 안 된다
    const ids = new Set(runs.data.map((r) => r.id))
    setCompare((c) => (c.every((id) => ids.has(id)) ? c : c.filter((id) => ids.has(id))))
    setView((v) => (v.kind === 'run' && !ids.has(v.id) ? { kind: 'new' } : v))
  }, [runs.status, runs.data])

  /*
   * 비교는 이제 사이드바 하단에서 직접 들어가는 화면이다. 체크박스도 이 화면에서만 나오므로
   * 고른 것이 없다고 화면을 튕기면 고를 방법 자체가 사라진다. 빈 상태는 CompareView 가 맡는다.
   */

  // 상세(dataset 포함)는 목록에 없는 정보다. 상태가 바뀔 때마다 다시 받는다.
  // run 을 빠르게 옮겨 다니면 이전 run 의 응답이 늦게 도착할 수 있어 취소 플래그를 둔다.
  useEffect(() => {
    if (!runId) {
      setDetail(null)
      return
    }
    let cancelled = false
    api
      .run(runId)
      .then((r) => !cancelled && setDetail(r))
      .catch(() => !cancelled && setDetail(null))
    return () => {
      cancelled = true
    }
  }, [runId, current?.status])

  async function stop(mode: 'graceful' | 'force') {
    if (!runId) return
    try {
      await api.stopRun(runId, mode)
      runs.reload()
    } catch (e) {
      toast(String(e instanceof Error ? e.message : e), 'error')
    }
  }

  async function deleteRun(run: Run) {
    const ok = await confirm({
      title: `'${run.name}' 을 삭제할까요?`,
      body: '가중치와 플롯을 포함한 산출물 폴더가 통째로 지워집니다. 되돌릴 수 없습니다.',
      confirmLabel: '삭제',
      danger: true,
    })
    if (!ok) return
    try {
      await api.deleteRun(run.id)
      // 불변식을 목록 재조회에 맡기지 않는다 — 재조회가 실패하거나 늦으면
      // 지워진 run 화면에 그대로 남는다. 지운 즉시 여기서 끊는다.
      setCompare((c) => c.filter((id) => id !== run.id))
      setView((v) => (v.kind === 'run' && v.id === run.id ? { kind: 'new' } : v))
      toast(`${run.name} 삭제됨`, 'success')
    } catch (e) {
      toast(String(e instanceof Error ? e.message : e), 'error')
    }
    runs.reload()
  }

  const offline = runs.failures >= OFFLINE_AFTER

  return (
    <div className="app">
      <header className="header">
        <h1>YOLO 학습 콘솔</h1>
        <div className="gpu-strip spacer">
          {gpus.data.map((g) => (
            <span key={g.index}>
              GPU{g.index}
              {/* 막대는 숫자를 대신하지 않고 옆에 붙는다 — 몇 %인지는 숫자가 답한다. */}
              <span className="meter" aria-hidden="true">
                <div style={{ width: `${Math.min(100, g.utilization)}%` }} />
              </span>
              {g.utilization}% {Math.round(g.memory_used_mb / 1024)}/{Math.round(g.memory_total_mb / 1024)}GB
            </span>
          ))}
        </div>
      </header>

      {offline && (
        <div className="banner" role="status">
          <span>서버에 연결하지 못하고 있습니다 ({runs.failures}회 연속 실패). 화면의 값이 오래된 것일 수 있습니다.</span>
          <button className="btn-xs spacer" onClick={() => runs.reload()}>
            다시 시도
          </button>
        </div>
      )}

      <div className="body">
        <Sidebar
          runs={runs.data}
          runsStatus={runs.status}
          onRetryRuns={() => runs.reload()}
          datasetCount={datasets.data.length}
          view={view}
          compare={compare}
          onNavigate={setView}
          onToggleCompare={(id, on) =>
            setCompare((c) => (on ? (c.length >= COMPARE_LIMIT ? c : [...c, id]) : c.filter((x) => x !== id)))
          }
          onClearCompare={() => setCompare([])}
          onDeleteRun={deleteRun}
        />

        <main className="main">
          {view.kind === 'run' && current && (
            <RunHeader run={current} dataset={detail?.dataset} stream={stream} onStop={stop} />
          )}
          {view.kind === 'run' && current?.status === 'failed' && (
            <FailureCard
              run={current}
              onStarted={(id) => {
                runs.reload()
                setView({ kind: 'run', id })
              }}
            />
          )}

          {view.kind === 'new' && (
            <NewRunPanel
              datasets={datasets.data}
              datasetsStatus={datasets.status}
              onRetryDatasets={() => datasets.reload()}
              gpus={gpus.data}
              gpusStatus={gpus.status}
              onRetryGpus={() => gpus.reload()}
              onRegisterDataset={() => setView({ kind: 'datasets' })}
              onStarted={(id) => {
                runs.reload()
                setView({ kind: 'run', id })
              }}
            />
          )}

          {view.kind === 'run' && runId && (
            <SplitPane
              storageKey="yolo.split.run"
              label="예측 화면과 지표 화면의 너비"
              left={
                <PreviewPanel
                  runId={runId}
                  run={current}
                  events={stream.events}
                  dataset={detail?.dataset}
                />
              }
              right={
                <div className="pane stack">
                  <MetricsChart events={stream.events} />
                  <LossChart events={stream.events} />
                  <LogView lines={stream.logs} />
                </div>
              }
            />
          )}

          {view.kind === 'datasets' && (
            <DatasetsView datasets={datasets.data} status={datasets.status} onChanged={() => datasets.reload()} />
          )}

          {view.kind === 'compare' && (
            <div className="pane">
              <CompareView runIds={compare} />
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
