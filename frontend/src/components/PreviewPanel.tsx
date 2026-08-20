import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type {
  Artifacts,
  Dataset,
  ExportStatus,
  PredictResult,
  Run,
  SystemInfo,
  TrainEvent,
} from '../types'
import { DatasetReviewPanel } from './DatasetReviewPanel'
import { QualityPanel } from './QualityPanel'
import { DiagnosePanel } from './DiagnosePanel'
import { Field } from './ui/Field'
import { EmptyState, SkeletonRows } from './ui/EmptyState'
import { TabPanel, Tabs } from './ui/Tabs'

const TABS = [
  { key: 'pred', label: '예측' },
  { key: 'plots', label: '플롯' },
  { key: 'diagnose', label: '진단' },
  { key: 'infer', label: '추론' },
  { key: 'dataset', label: '데이터셋' },
] as const

const ID_PREFIX = 'preview'

interface Props {
  runId: string
  /** 진단 탭이 상태와 데이터셋 참조를 쓴다. 목록이 아직 안 왔으면 null 이다. */
  run: Run | null
  events: TrainEvent[]
  dataset: Dataset | null | undefined
}

export function PreviewPanel({ runId, run, events, dataset }: Props) {
  const [tab, setTab] = useState<string>('pred')
  const [artifacts, setArtifacts] = useState<Artifacts | null>(null)
  const finished = events.some((e) => e.t === 'end')

  useEffect(() => {
    setArtifacts(null)
    api.artifacts(runId).then(setArtifacts).catch(() => setArtifacts(null))
  }, [runId, finished])

  return (
    <>
      <Tabs items={TABS} value={tab} onChange={setTab} label="실행 상세 보기" idPrefix={ID_PREFIX} />
      <TabPanel idPrefix={ID_PREFIX} tabKey={tab} className="pane">
        {tab === 'pred' && <EpochPreview runId={runId} events={events} />}
        {tab === 'plots' && <Plots runId={runId} artifacts={artifacts} finished={finished} />}
        {tab === 'diagnose' &&
          (run ? (
            <DiagnosePanel run={run} dataset={dataset} />
          ) : (
            <EmptyState title="실행 정보를 불러오는 중입니다." />
          ))}
        {tab === 'infer' && <InferenceTest runId={runId} />}
        {tab === 'dataset' && (
          <>
            <DatasetReviewPanel dataset={dataset} />
            <QualityPanel dataset={dataset} />
          </>
        )}
      </TabPanel>
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

  // 드래그가 끊기지 않도록 인접 프레임을 미리 받아둔다. 화면에 뜨는 것과 같은 파일이어야 의미가 있다.
  useEffect(() => {
    for (const offset of [-1, 1]) {
      const files = frames[index + offset]?.[1]
      if (!files?.length) continue
      const img = new Image()
      img.src = api.fileUrl(runId, files.find((f) => f.includes('_pred')) ?? files[0])
    }
  }, [frames, index, runId])

  if (!frames.length) {
    return <EmptyState title="아직 검증 예측 이미지가 없습니다" description="첫 에폭의 검증이 끝나면 여기에 나타납니다." />
  }

  const [epoch, files] = frames[Math.min(index, frames.length - 1)]
  const pred = files.find((f) => f.includes('_pred')) ?? files[0]
  const gt = files.find((f) => f.includes('_labels'))

  return (
    <>
      <div className="row wrap" style={{ marginBottom: 10 }}>
        <strong>{epoch} 에폭</strong>
        <span className="muted small">{frames.length}개 에폭 기록됨</span>
        <label className="row tight small muted spacer nowrap">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          최신 따라가기
        </label>
        {gt && (
          <label className="row tight small muted nowrap">
            <input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} />
            정답 대조
          </label>
        )}
      </div>

      <input
        className="slider"
        type="range"
        aria-label="에폭 선택"
        aria-valuetext={`${epoch} 에폭`}
        min={0}
        max={frames.length - 1}
        value={Math.min(index, frames.length - 1)}
        onChange={(e) => {
          setFollow(false)
          setIndex(Number(e.target.value))
        }}
      />

      <div style={{ display: 'grid', gap: 10, gridTemplateColumns: showGt && gt ? '1fr 1fr' : '1fr', marginTop: 10 }}>
        <figure className="preview-stage" style={{ margin: 0 }}>
          <img className="preview-img" src={api.fileUrl(runId, pred)} alt={`${epoch} 에폭 예측`} />
          <figcaption className="preview-name">{fileName(pred)}</figcaption>
        </figure>
        {showGt && gt && (
          <figure className="preview-stage" style={{ margin: 0 }}>
            <img className="preview-img" src={api.fileUrl(runId, gt)} alt={`${epoch} 에폭 정답`} />
            <figcaption className="preview-name">{fileName(gt)}</figcaption>
          </figure>
        )}
      </div>

      {/*
        에폭 썸네일. 100에폭이면 100장이라 전부 받으면 안 된다 — loading="lazy" 로 보이는
        것만 받게 두고, 가로 스크롤로 나머지에 닿는다.
      */}
      <div className="thumb-strip" role="group" aria-label="에폭 썸네일">
        {frames.map(([e, fs], i) => {
          const thumb = fs.find((f) => f.includes('_pred')) ?? fs[0]
          const on = i === Math.min(index, frames.length - 1)
          return (
            <button
              key={e}
              className={on ? 'on' : undefined}
              aria-label={`${e} 에폭`}
              aria-current={on ? 'true' : undefined}
              onClick={() => {
                setFollow(false)
                setIndex(i)
              }}
            >
              <img src={api.fileUrl(runId, thumb)} alt="" loading="lazy" />
            </button>
          )
        })}
      </div>
    </>
  )
}

/** 경로에서 파일 이름만. 칩에 전체 경로를 넣으면 이미지 절반을 덮는다. */
function fileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

function Plots({ runId, artifacts, finished }: { runId: string; artifacts: Artifacts | null; finished: boolean }) {
  if (!artifacts) return <SkeletonRows rows={3} />
  if (!artifacts.plots.length && !artifacts.weights.length) {
    return (
      <EmptyState
        title={finished ? '생성된 플롯이 없습니다' : '아직 산출물이 없습니다'}
        description={finished ? undefined : '학습이 끝나면 플롯이 생성됩니다.'}
      />
    )
  }
  return (
    <>
      {artifacts.weights.length > 0 && (
        <div className="card">
          <h3>가중치</h3>
          <div className="row wrap">
            {artifacts.weights.map((w) => (
              <a key={w} className="button btn-sm" href={api.fileUrl(runId, w)} download>
                {w.split('/').pop()} 내려받기
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
      <div className="small muted" style={{ marginBottom: 6 }}>
        다른 포맷으로 내보내기
      </div>
      <div className="row wrap" style={{ alignItems: 'flex-end' }}>
        <Field label="포맷">
          {(props) => (
            <select {...props} style={{ width: 170 }} value={format} onChange={(e) => setFormat(e.target.value)}>
              {formats.map((f) => (
                <option key={f.value} value={f.value} disabled={!f.ok}>
                  {f.label}
                  {f.ok ? '' : ' (미설치)'}
                </option>
              ))}
            </select>
          )}
        </Field>
        <Field label="가중치">
          {(props) => (
            <select {...props} style={{ width: 150 }} value={weight} onChange={(e) => setWeight(e.target.value)}>
              {weights.map((w) => (
                <option key={w} value={w}>
                  {w.split('/').pop()}
                </option>
              ))}
            </select>
          )}
        </Field>
        <Field label="imgsz">
          {(props) => (
            <input
              {...props}
              type="number"
              style={{ width: 90 }}
              min={32}
              max={4096}
              step={32}
              value={imgsz}
              onChange={(e) => setImgsz(Number(e.target.value))}
            />
          )}
        </Field>
        <label className="row tight small muted nowrap" style={{ paddingBottom: 6 }}>
          <input type="checkbox" checked={half} onChange={(e) => setHalf(e.target.checked)} />
          FP16
        </label>
        <button className="primary" disabled={running || !weight} onClick={start} style={{ marginBottom: 2 }}>
          {running ? '변환 중…' : '내보내기'}
        </button>
      </div>

      {error && (
        <div className="error small" style={{ marginTop: 8 }}>
          {error}
        </div>
      )}

      {result?.status === 'completed' && result.file && (
        <div style={{ marginTop: 8 }}>
          <a className="button btn-sm" href={api.fileUrl(runId, result.file)} download>
            {result.file.split('/').pop()} 내려받기 ({result.size_mb} MB)
          </a>
        </div>
      )}
      {result?.status === 'failed' && (
        <div className="log mono error" style={{ marginTop: 8, maxHeight: 120, flex: 'none' }}>
          {result.error}
        </div>
      )}
    </div>
  )
}

/** 학습된 가중치로 임의 이미지에 추론해 본다. 서버가 CPU 를 강제하므로 학습과 경합하지 않는다. */
function InferenceTest({ runId }: { runId: string }) {
  const [weights, setWeights] = useState<{ value: string; label: string; size_mb: number }[]>([])
  const [selected, setSelected] = useState('')
  /*
   * conf 기본값은 진단이 잰 값을 쓴다. 0.25 는 ultralytics 의 관례일 뿐 이 모델에 맞는
   * 값이 아니다. 진단을 돌리지 않았거나 서버가 "믿을 수 없다"(reliable=false)고 하면
   * 0.25 로 남기고 근거 줄도 띄우지 않는다 — 재보지 않은 값을 권하지 않는다.
   */
  const [conf, setConf] = useState(0.25)
  const [confAdvice, setConfAdvice] = useState<{ conf: number; f1: number | null } | null>(null)
  const [confTouched, setConfTouched] = useState(false)
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

  useEffect(() => {
    let cancelled = false
    api
      .analysisReport(runId)
      .then((r) => {
        const c = r.conf_recommendation
        if (cancelled || !c?.reliable || c.conf == null) return
        setConfAdvice({ conf: c.conf, f1: c.f1 })
        // 사용자가 이미 슬라이더를 만졌으면 덮어쓰지 않는다.
        setConf((v) => (confTouched ? v : c.conf!))
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    return <EmptyState title="아직 저장된 가중치가 없습니다" description="첫 에폭이 끝나면 추론할 수 있습니다." />
  }

  return (
    <div className="infer-grid">
      <div className="card">
        <h3>추론 설정</h3>
        <div className="stack">
          <Field label="가중치">
            {(props) => (
              <select {...props} value={selected} onChange={(e) => setSelected(e.target.value)}>
                {weights.map((w) => (
                  <option key={w.value} value={w.value}>
                    {w.label} ({w.size_mb} MB)
                  </option>
                ))}
              </select>
            )}
          </Field>
          <div>
            <RangeField
              label="확신도 임계값 conf"
              value={conf}
              min={0.01}
              max={0.95}
              step={0.01}
              onChange={(v) => {
                setConfTouched(true)
                setConf(v)
              }}
            />
            {confAdvice && (
              <div className="help">
                진단이 이 모델에서 잰 값입니다 — {confAdvice.conf} 에서 F1 이 가장 높았습니다
                {confAdvice.f1 != null && ` (${(confAdvice.f1 * 100).toFixed(1)}%)`}.
              </div>
            )}
          </div>
          <RangeField label="NMS IoU" value={iou} min={0.1} max={0.95} step={0.05} onChange={setIou} />
          <Field label="이미지 크기 imgsz">
            {(props) => (
              <input
                {...props}
                type="number"
                min={32}
                max={4096}
                step={32}
                value={imgsz}
                onChange={(e) => setImgsz(Number(e.target.value))}
              />
            )}
          </Field>
        </div>
        <div className="help">추론은 항상 CPU 에서 실행됩니다 — 학습 중인 GPU 와 경합하지 않기 위해서입니다.</div>
      </div>

      <div>
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
      >
        <button type="button" disabled={busy} onClick={() => fileRef.current?.click()}>
          {busy ? '추론 중…' : '이미지를 끌어다 놓거나 클릭해서 선택'}
        </button>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => e.target.files?.[0] && predict(e.target.files[0])}
      />

      {lastFile.current && !busy && (
        <button className="btn-sm" style={{ marginTop: 8 }} onClick={() => lastFile.current && predict(lastFile.current)}>
          같은 이미지로 다시 (설정 변경 반영)
        </button>
      )}

      {error && (
        <div className="card error small" style={{ marginTop: 10 }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 12 }}>
          <img className="preview-img" src={api.fileUrl(runId, result.image)} alt="추론 결과" />
          <div className="row small muted wrap" style={{ marginTop: 6, gap: 12 }}>
            <span>검출 {result.count}개</span>
            <span>{result.elapsed_ms} ms</span>
            <span>{result.weights}</span>
          </div>
          {result.detections.length > 0 && (
            <table style={{ marginTop: 8 }}>
              <caption className="sr-only">검출된 객체 목록</caption>
              <thead>
                <tr>
                  <th scope="col">클래스</th>
                  <th scope="col">확신도</th>
                  <th scope="col">박스 (x1, y1, x2, y2)</th>
                </tr>
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
      </div>
    </div>
  )
}

function RangeField({
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
    <Field label={label} labelExtra={<span className="mono muted"> {value}</span>}>
      {(props) => (
        <input
          {...props}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      )}
    </Field>
  )
}
