import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { Artifacts, Dataset, TrainEvent } from '../types'

type Tab = '예측' | '플롯' | '데이터셋'

interface Props {
  runId: string
  events: TrainEvent[]
  dataset: Dataset | null | undefined
}

export function PreviewPanel({ runId, events, dataset }: Props) {
  const [tab, setTab] = useState<Tab>('예측')
  const [artifacts, setArtifacts] = useState<Artifacts | null>(null)
  const finished = events.some((e) => e.t === 'end')

  useEffect(() => {
    setArtifacts(null)
    api.artifacts(runId).then(setArtifacts).catch(() => setArtifacts(null))
  }, [runId, finished])

  return (
    <>
      <div className="tabs">
        {(['예측', '플롯', '데이터셋'] as Tab[]).map((t) => (
          <div key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t}
          </div>
        ))}
      </div>
      <div className="pane">
        {tab === '예측' && <EpochPreview runId={runId} events={events} />}
        {tab === '플롯' && <Plots runId={runId} artifacts={artifacts} finished={finished} />}
        {tab === '데이터셋' && <DatasetSamples dataset={dataset} />}
      </div>
    </>
  )
}

/**
 * 에폭별 검증 예측 이미지 + 스크러버.
 *
 * 실시간 갱신만 하면 지나간 에폭을 비교할 수 없다. 슬라이더로 1에폭과 마지막 에폭을
 * 오가며 박스가 정확해지는 과정을 직접 볼 수 있게 하는 것이 이 화면의 핵심이다.
 */
function EpochPreview({ runId, events }: { runId: string; events: TrainEvent[] }) {
  const [follow, setFollow] = useState(true)
  const [index, setIndex] = useState(0)
  const [showGt, setShowGt] = useState(false)

  const frames = useMemo(() => {
    const byEpoch = new Map<number, string[]>()
    for (const e of events) {
      if (e.t === 'artifact' && e.epoch != null && e.files?.length) byEpoch.set(e.epoch, e.files)
    }
    return [...byEpoch.entries()].sort((a, b) => a[0] - b[0])
  }, [events])

  useEffect(() => {
    if (follow && frames.length) setIndex(frames.length - 1)
  }, [frames.length, follow])

  // 드래그가 끊기지 않도록 인접 프레임을 미리 받아둔다.
  useEffect(() => {
    for (const offset of [-1, 1]) {
      const frame = frames[index + offset]
      if (frame) {
        const img = new Image()
        img.src = api.fileUrl(runId, frame[1][0] ?? frame[1][0])
      }
    }
  }, [frames, index, runId])

  if (!frames.length) {
    return (
      <p className="muted">
        아직 검증 예측 이미지가 없습니다. 첫 에폭의 검증이 끝나면 여기에 나타납니다.
      </p>
    )
  }

  const [epoch, files] = frames[Math.min(index, frames.length - 1)]
  const pred = files.find((f) => f.includes('_pred')) ?? files[0]
  const gt = files.find((f) => f.includes('_labels'))

  return (
    <>
      <div className="row" style={{ marginBottom: 10, gap: 12 }}>
        <strong>{epoch} 에폭</strong>
        <span className="muted small">{frames.length}개 에폭 기록됨</span>
        <label className="row small muted" style={{ gap: 4, marginLeft: 'auto' }}>
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          최신 따라가기
        </label>
        {gt && (
          <label className="row small muted" style={{ gap: 4 }}>
            <input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} />
            정답 대조
          </label>
        )}
      </div>

      <input
        className="slider"
        type="range"
        min={0}
        max={frames.length - 1}
        value={Math.min(index, frames.length - 1)}
        onChange={(e) => {
          setFollow(false)
          setIndex(Number(e.target.value))
        }}
      />

      <div style={{ display: 'grid', gap: 10, gridTemplateColumns: showGt && gt ? '1fr 1fr' : '1fr', marginTop: 10 }}>
        <figure style={{ margin: 0 }}>
          <img className="preview-img" src={api.fileUrl(runId, pred)} alt={`${epoch} 에폭 예측`} />
          <figcaption className="small muted">예측</figcaption>
        </figure>
        {showGt && gt && (
          <figure style={{ margin: 0 }}>
            <img className="preview-img" src={api.fileUrl(runId, gt)} alt={`${epoch} 에폭 정답`} />
            <figcaption className="small muted">정답(GT)</figcaption>
          </figure>
        )}
      </div>
    </>
  )
}

function Plots({ runId, artifacts, finished }: { runId: string; artifacts: Artifacts | null; finished: boolean }) {
  if (!artifacts) return <p className="muted">불러오는 중…</p>
  if (!artifacts.plots.length) {
    return <p className="muted">{finished ? '생성된 플롯이 없습니다.' : '학습이 끝나면 플롯이 생성됩니다.'}</p>
  }
  return (
    <>
      {artifacts.weights.length > 0 && (
        <div className="card">
          <h3>가중치</h3>
          <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
            {artifacts.weights.map((w) => (
              <a key={w} href={api.fileUrl(runId, w)} download>
                <button>{w.split('/').pop()} 내려받기</button>
              </a>
            ))}
          </div>
        </div>
      )}
      <div className="gallery">
        {artifacts.plots.map((p) => (
          <figure key={p}>
            <a href={api.fileUrl(runId, p)} target="_blank" rel="noreferrer">
              <img className="preview-img" src={api.fileUrl(runId, p)} alt={p} />
            </a>
            <figcaption>{p.split('/').pop()}</figcaption>
          </figure>
        ))}
      </div>
    </>
  )
}

function DatasetSamples({ dataset }: { dataset: Dataset | null | undefined }) {
  const [samples, setSamples] = useState<{ path: string; boxes: { name: string; cx: number; cy: number; w: number; h: number }[] }[]>([])

  useEffect(() => {
    if (!dataset) return
    api.datasetSamples(dataset.id).then((r) => setSamples(r.samples)).catch(() => setSamples([]))
  }, [dataset?.id])

  if (!dataset) return <p className="muted">데이터셋 정보가 없습니다.</p>

  const report = dataset.report
  return (
    <>
      <div className="card">
        <h3>검수 요약</h3>
        <table>
          <tbody>
            <tr><th>이미지</th><td>{report.total_images.toLocaleString()}장</td></tr>
            <tr><th>train / val</th><td>{report.train_count} / {report.val_count} {report.auto_split && <span className="muted">(자동 분할)</span>}</td></tr>
            <tr><th>클래스</th><td>{dataset.classes.join(', ')}</td></tr>
            <tr><th>라벨 없는 이미지</th><td className={report.missing_labels.length ? 'error' : ''}>{report.missing_labels.length}건</td></tr>
            <tr><th>좌표 이상</th><td className={report.label_issues.length ? 'error' : ''}>{report.label_issues.length}건</td></tr>
          </tbody>
        </table>
        {report.class_instances && (
          <div style={{ marginTop: 10 }}>
            {Object.entries(report.class_instances).map(([name, count]) => {
              const max = Math.max(...Object.values(report.class_instances!))
              return (
                <div key={name} className="row small" style={{ gap: 8, marginBottom: 4 }}>
                  <span style={{ width: 90 }} className="muted">{name}</span>
                  <div className="progress"><div style={{ width: `${(count / max) * 100}%` }} /></div>
                  <span style={{ width: 60, textAlign: 'right' }}>{count.toLocaleString()}</span>
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div className="gallery">
        {samples.map((s) => (
          <figure key={s.path}>
            <SampleImage datasetId={dataset.id} path={s.path} boxes={s.boxes} />
            <figcaption>{s.path.split(/[\\/]/).pop()} · {s.boxes.length}개 박스</figcaption>
          </figure>
        ))}
      </div>
    </>
  )
}

function SampleImage({
  datasetId,
  path,
  boxes,
}: {
  datasetId: string
  path: string
  boxes: { name: string; cx: number; cy: number; w: number; h: number }[]
}) {
  const ref = useRef<HTMLDivElement>(null)
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <img className="preview-img" src={api.datasetImageUrl(datasetId, path)} alt={path} />
      {boxes.map((b, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: `${(b.cx - b.w / 2) * 100}%`,
            top: `${(b.cy - b.h / 2) * 100}%`,
            width: `${b.w * 100}%`,
            height: `${b.h * 100}%`,
            border: '1.5px solid #35c46b',
            borderRadius: 2,
          }}
          title={b.name}
        />
      ))}
    </div>
  )
}
