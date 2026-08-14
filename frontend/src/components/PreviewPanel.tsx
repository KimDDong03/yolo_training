import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { Artifacts, Dataset, ExportStatus, PredictResult, SystemInfo, TrainEvent } from '../types'
import { DatasetReviewPanel } from './DatasetReviewPanel'

type Tab = '예측' | '플롯' | '추론' | '데이터셋'

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
        {(['예측', '플롯', '추론', '데이터셋'] as Tab[]).map((t) => (
          <div key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t}
          </div>
        ))}
      </div>
      <div className="pane">
        {tab === '예측' && <EpochPreview runId={runId} events={events} />}
        {tab === '플롯' && <Plots runId={runId} artifacts={artifacts} finished={finished} />}
        {tab === '추론' && <InferenceTest runId={runId} />}
        {tab === '데이터셋' && <DatasetReviewPanel dataset={dataset} />}
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
          <ExportPanel runId={runId} weights={artifacts.weights} />
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

/**
 * 가중치를 다른 포맷으로 변환한다.
 *
 * TensorRT 는 수 분 걸리고 실패도 잦다 — 그래서 에러 메시지를 감추지 않고 그대로 보여준다.
 * 상태는 WebSocket 이 아니라 폴링으로 본다. 이벤트가 두어 개뿐이라 스트림을 열 이유가 없다.
 */
function ExportPanel({ runId, weights }: { runId: string; weights: string[] }) {
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [status, setStatus] = useState<ExportStatus | null>(null)
  const [format, setFormat] = useState('onnx')
  const [weight, setWeight] = useState(weights[0] ?? '')
  const [imgsz, setImgsz] = useState(640)
  const [half, setHalf] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.systemInfo().then(setInfo).catch(() => {})
    api.exportStatus(runId).then(setStatus).catch(() => {})
  }, [runId])

  useEffect(() => {
    if (status?.status !== 'running') return
    const timer = setInterval(() => {
      api.exportStatus(runId).then(setStatus).catch(() => {})
    }, 2000)
    return () => clearInterval(timer)
  }, [status?.status, runId])

  const formats = [
    { value: 'onnx', label: 'ONNX', ok: info?.onnx !== false },
    { value: 'torchscript', label: 'TorchScript', ok: true },
    { value: 'engine', label: 'TensorRT (GPU 필요)', ok: info?.tensorrt !== false },
  ]

  async function start() {
    setError('')
    try {
      setStatus(await api.startExport(runId, { format, weights: weight, imgsz, half }))
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    }
  }

  const running = status?.status === 'running'
  const result = status?.result

  return (
    <div style={{ marginTop: 12, borderTop: '1px solid var(--line)', paddingTop: 10 }}>
      <div className="small muted" style={{ marginBottom: 6 }}>다른 포맷으로 내보내기</div>
      <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
        <select style={{ width: 170 }} value={format} onChange={(e) => setFormat(e.target.value)}>
          {formats.map((f) => (
            <option key={f.value} value={f.value} disabled={!f.ok}>
              {f.label}{f.ok ? '' : ' (미설치)'}
            </option>
          ))}
        </select>
        <select style={{ width: 150 }} value={weight} onChange={(e) => setWeight(e.target.value)}>
          {weights.map((w) => (
            <option key={w} value={w}>{w.split('/').pop()}</option>
          ))}
        </select>
        <input
          type="number"
          style={{ width: 90 }}
          min={32}
          max={4096}
          step={32}
          value={imgsz}
          onChange={(e) => setImgsz(Number(e.target.value))}
        />
        <label className="row small muted" style={{ gap: 4 }}>
          <input type="checkbox" checked={half} onChange={(e) => setHalf(e.target.checked)} />
          FP16
        </label>
        <button className="primary" disabled={running || !weight} onClick={start}>
          {running ? '변환 중…' : '내보내기'}
        </button>
      </div>

      {error && <div className="error small" style={{ marginTop: 8 }}>{error}</div>}

      {result?.status === 'completed' && result.file && (
        <div className="small" style={{ marginTop: 8 }}>
          <a href={api.fileUrl(runId, result.file)} download>
            <button>{result.file.split('/').pop()} 내려받기 ({result.size_mb} MB)</button>
          </a>
        </div>
      )}
      {result?.status === 'failed' && (
        <div className="log mono error" style={{ marginTop: 8, maxHeight: 120 }}>{result.error}</div>
      )}
    </div>
  )
}

/** 학습된 가중치로 임의 이미지에 추론해 본다. 서버가 CPU 를 강제하므로 학습과 경합하지 않는다. */
function InferenceTest({ runId }: { runId: string }) {
  const [weights, setWeights] = useState<{ value: string; label: string; size_mb: number }[]>([])
  const [selected, setSelected] = useState('')
  const [conf, setConf] = useState(0.25)
  const [iou, setIou] = useState(0.7)
  const [imgsz, setImgsz] = useState(640)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [over, setOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const lastFile = useRef<File | null>(null)

  useEffect(() => {
    api
      .runWeights(runId)
      .then((r) => {
        setWeights(r.weights)
        setSelected((s) => s || r.weights.find((w) => w.label === 'best')?.value || r.weights[0]?.value || '')
      })
      .catch(() => setWeights([]))
  }, [runId])

  async function predict(file: File) {
    lastFile.current = file
    setBusy(true)
    setError('')
    try {
      setResult(await api.predict(runId, file, { weights: selected, conf, iou, imgsz }))
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  if (!weights.length) {
    return <p className="muted">아직 저장된 가중치가 없습니다. 첫 에폭이 끝나면 추론할 수 있습니다.</p>
  }

  return (
    <>
      <div className="card">
        <h3>추론 설정</h3>
        <div className="grid">
          <div className="field">
            <label>가중치</label>
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              {weights.map((w) => (
                <option key={w.value} value={w.value}>
                  {w.label} ({w.size_mb} MB)
                </option>
              ))}
            </select>
          </div>
          <Slider label="확신도 임계값 conf" value={conf} min={0.01} max={0.95} step={0.01} onChange={setConf} />
          <Slider label="NMS IoU" value={iou} min={0.1} max={0.95} step={0.05} onChange={setIou} />
          <div className="field">
            <label>이미지 크기 imgsz</label>
            <input type="number" min={32} max={4096} step={32} value={imgsz} onChange={(e) => setImgsz(Number(e.target.value))} />
          </div>
        </div>
        <div className="help">추론은 항상 CPU 에서 실행됩니다 — 학습 중인 GPU 와 경합하지 않기 위해서입니다.</div>
      </div>

      <div
        className={`drop ${over ? 'over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          const file = e.dataTransfer.files[0]
          if (file) predict(file)
        }}
        onClick={() => fileRef.current?.click()}
      >
        {busy ? '추론 중…' : '이미지를 끌어다 놓거나 클릭해서 선택'}
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && predict(e.target.files[0])}
        />
      </div>

      {lastFile.current && !busy && (
        <button style={{ marginTop: 8, fontSize: 12 }} onClick={() => lastFile.current && predict(lastFile.current)}>
          같은 이미지로 다시 (설정 변경 반영)
        </button>
      )}

      {error && <div className="card error small" style={{ marginTop: 10 }}>{error}</div>}

      {result && (
        <div style={{ marginTop: 12 }}>
          <img className="preview-img" src={api.fileUrl(runId, result.image)} alt="추론 결과" />
          <div className="row small muted" style={{ marginTop: 6, gap: 12 }}>
            <span>검출 {result.count}개</span>
            <span>{result.elapsed_ms} ms</span>
            <span>{result.weights}</span>
          </div>
          {result.detections.length > 0 && (
            <table style={{ marginTop: 8 }}>
              <thead>
                <tr><th>클래스</th><th>확신도</th><th>박스 (x1, y1, x2, y2)</th></tr>
              </thead>
              <tbody>
                {result.detections.map((d, i) => (
                  <tr key={i}>
                    <td>{d.name}</td>
                    <td>{(d.conf * 100).toFixed(1)}%</td>
                    <td className="mono muted">{d.xyxy.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </>
  )
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <div className="field">
      <label>
        {label} <span className="mono muted">{value}</span>
      </label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  )
}
