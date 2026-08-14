import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { Dataset, Gpu, ParamField, ParamSchema } from '../types'

interface Props {
  datasets: Dataset[]
  gpus: Gpu[]
  onDatasetsChanged: () => void
  onStarted: (runId: string) => void
}

export function NewRunPanel({ datasets, gpus, onDatasetsChanged, onStarted }: Props) {
  const [schema, setSchema] = useState<ParamSchema | null>(null)
  const [presets, setPresets] = useState<Record<string, Record<string, unknown>>>({})
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [datasetId, setDatasetId] = useState('')
  const [devices, setDevices] = useState<number[]>([])
  const [advanced, setAdvanced] = useState(false)
  const [filter, setFilter] = useState('')
  const [runName, setRunName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.paramsSchema().then((r) => {
      setSchema(r.schema)
      setPresets(r.presets)
      setValues(Object.fromEntries(r.schema.fields.map((f) => [f.key, f.default])))
    })
  }, [])

  useEffect(() => {
    if (!datasetId && datasets.length) setDatasetId(datasets[0].id)
  }, [datasets, datasetId])

  useEffect(() => {
    if (!devices.length && gpus.length) setDevices([gpus[0].index])
  }, [gpus])

  const dataset = datasets.find((d) => d.id === datasetId)
  const groups = useMemo(() => {
    if (!schema) return []
    const needle = filter.trim().toLowerCase()
    return schema.groups
      .map((group) => ({
        group,
        fields: schema.fields.filter(
          (f) =>
            f.group === group &&
            (advanced || !f.advanced) &&
            (!needle || f.key.toLowerCase().includes(needle) || f.label.toLowerCase().includes(needle)),
        ),
      }))
      .filter((g) => g.fields.length > 0)
  }, [schema, advanced, filter])

  const cli = useMemo(() => {
    if (!schema || !dataset) return ''
    const parts = ['yolo', 'train', `data="${dataset.yaml_path}"`]
    for (const field of schema.fields) {
      const v = values[field.key]
      if (v === field.default || v === undefined || v === null || v === '') continue
      parts.push(`${field.key}=${typeof v === 'string' && v.includes(' ') ? `"${v}"` : v}`)
    }
    if (values['model']) parts.push(`model="${values['model']}"`)
    parts.push(`device=${devices.length ? devices.join(',') : 'cpu'}`)
    return parts.join(' ')
  }, [schema, values, dataset, devices])

  async function start() {
    if (!dataset) return
    setBusy(true)
    setError('')
    try {
      const run = await api.createRun({
        dataset_id: dataset.id,
        name: runName || dataset.name,
        devices,
        params: values,
      })
      onStarted(run.id)
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="pane">
      <DatasetRegister onDone={onDatasetsChanged} />

      <div className="card">
        <h3>학습 대상</h3>
        <div className="grid">
          <div className="field">
            <label>데이터셋</label>
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
              {datasets.length === 0 && <option value="">등록된 데이터셋이 없습니다</option>}
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} · {d.report.total_images}장 · {d.classes.length}클래스
                </option>
              ))}
            </select>
            {dataset && <div className="help mono">{dataset.yaml_path}</div>}
          </div>
          <div className="field">
            <label>실행 이름</label>
            <input value={runName} placeholder={dataset?.name ?? ''} onChange={(e) => setRunName(e.target.value)} />
          </div>
        </div>

        <div className="field" style={{ marginTop: 10 }}>
          <label>사용할 GPU</label>
          {gpus.length === 0 ? (
            <div className="small muted">GPU를 찾지 못했습니다. CPU로 학습합니다(매우 느립니다).</div>
          ) : (
            <div className="row" style={{ flexWrap: 'wrap', gap: 12 }}>
              {gpus.map((g) => (
                <label key={g.index} className="row small" style={{ gap: 5 }}>
                  <input
                    type="checkbox"
                    checked={devices.includes(g.index)}
                    onChange={(e) =>
                      setDevices((d) =>
                        e.target.checked ? [...d, g.index].sort((a, b) => a - b) : d.filter((x) => x !== g.index),
                      )
                    }
                  />
                  #{g.index} {g.name} · {Math.round((g.memory_total_mb - g.memory_used_mb) / 1024)}GB 여유
                </label>
              ))}
            </div>
          )}
          {devices.length > 1 && (
            <div className="help">GPU를 2장 이상 고르면 ultralytics가 DDP(분산 학습)로 실행합니다.</div>
          )}
        </div>
      </div>

      <div className="card">
        <h3 className="row" style={{ gap: 10 }}>
          파라미터
          <input
            style={{ width: 160, marginLeft: 'auto' }}
            placeholder="이름으로 필터"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <label className="row small muted" style={{ gap: 4 }}>
            <input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)} />
            고급
          </label>
        </h3>

        <div className="row" style={{ gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
          <span className="small muted">프리셋</span>
          {Object.entries(presets).map(([name, patch]) => (
            <button key={name} style={{ fontSize: 12, padding: '3px 10px' }} onClick={() => setValues((v) => ({ ...v, ...patch }))}>
              {name}
            </button>
          ))}
        </div>

        {groups.map(({ group, fields }) => (
          <div key={group} style={{ marginBottom: 14 }}>
            <div className="small muted" style={{ marginBottom: 6 }}>{group}</div>
            <div className="grid">
              {fields.map((f) => (
                <Field key={f.key} field={f} value={values[f.key]} onChange={(v) => setValues((s) => ({ ...s, [f.key]: v }))} />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <h3>동등한 CLI 명령</h3>
        <div className="log mono" style={{ maxHeight: 90 }}>{cli}</div>
        <button style={{ marginTop: 8, fontSize: 12 }} onClick={() => navigator.clipboard?.writeText(cli)}>
          복사
        </button>
      </div>

      {error && <div className="card error">{error}</div>}

      <button className="primary" disabled={!dataset || busy} onClick={start} style={{ width: '100%', padding: 10 }}>
        {busy ? '시작하는 중…' : '학습 시작'}
      </button>
    </div>
  )
}

function Field({ field, value, onChange }: { field: ParamField; value: unknown; onChange: (v: unknown) => void }) {
  return (
    <div className="field">
      <label title={field.key}>
        {field.label} <span className="mono muted" style={{ fontSize: 10 }}>{field.key}</span>
      </label>
      {field.type === 'bool' ? (
        <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />
      ) : field.type === 'enum' && field.choices ? (
        <select value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>
          {field.choices.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          type="number"
          value={value === null || value === undefined ? '' : String(value)}
          min={field.min ?? undefined}
          max={field.max ?? undefined}
          step={field.step ?? undefined}
          onChange={(e) => {
            const raw = e.target.value
            if (raw === '') return onChange(null)
            onChange(field.type === 'int' ? parseInt(raw, 10) : parseFloat(raw))
          }}
        />
      )}
      {field.help && <div className="help">{field.help}</div>}
    </div>
  )
}

function DatasetRegister({ onDone }: { onDone: () => void }) {
  const [path, setPath] = useState('')
  const [valRatio, setValRatio] = useState(0.2)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [over, setOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  async function upload(file: File) {
    setBusy(true)
    setError('')
    try {
      await api.uploadZip(file, file.name.replace(/\.zip$/i, ''), valRatio)
      onDone()
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  async function register() {
    setBusy(true)
    setError('')
    try {
      await api.registerPath(path, path.split(/[\\/]/).filter(Boolean).pop() ?? path, valRatio)
      setPath('')
      onDone()
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h3>데이터셋 등록</h3>
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
          if (file) upload(file)
        }}
        onClick={() => fileRef.current?.click()}
      >
        {busy ? '처리 중…' : 'zip 파일을 여기에 끌어다 놓거나 클릭해서 선택'}
        <input
          ref={fileRef}
          type="file"
          accept=".zip"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
        />
      </div>

      <div className="row" style={{ gap: 8, marginTop: 10 }}>
        <input
          placeholder="또는 서버 폴더 경로 지정 — 예: D:\data\coins"
          value={path}
          onChange={(e) => setPath(e.target.value)}
        />
        <button disabled={!path || busy} onClick={register} style={{ whiteSpace: 'nowrap' }}>
          경로 등록
        </button>
      </div>
      <div className="row small muted" style={{ gap: 8, marginTop: 8 }}>
        <span>train/val 이 없을 때 검증 비율</span>
        <input
          type="number"
          step={0.05}
          min={0.05}
          max={0.5}
          value={valRatio}
          onChange={(e) => setValRatio(parseFloat(e.target.value))}
          style={{ width: 80 }}
        />
        <span>경로 등록은 파일을 복사하지 않고 원본을 그대로 참조합니다.</span>
      </div>
      {error && <div className="error small" style={{ marginTop: 8 }}>{error}</div>}
    </div>
  )
}
