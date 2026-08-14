import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api'
import type { Dataset, Gpu, ParamField, ParamSchema, Preset, SystemInfo, WeightCandidate } from '../types'
import type { LoadStatus } from '../useResource'
import { DatasetRegister } from './DatasetRegister'
import { DatasetReviewPanel } from './DatasetReviewPanel'
import { useConfirm, usePrompt } from './ui/Dialog'
import { Recommendations } from './Recommendations'
import { Field, type FieldStatus } from './ui/Field'
import { useToast } from './ui/Toast'

interface Props {
  datasets: Dataset[]
  datasetsStatus: LoadStatus
  onRetryDatasets: () => void
  gpus: Gpu[]
  gpusStatus: LoadStatus
  onRetryGpus: () => void
  onDatasetsChanged: () => void
  onStarted: (runId: string) => void
}

export function NewRunPanel({
  datasets,
  datasetsStatus,
  onRetryDatasets,
  gpus,
  gpusStatus,
  onRetryGpus,
  onDatasetsChanged,
  onStarted,
}: Props) {
  const [schema, setSchema] = useState<ParamSchema | null>(null)
  const [presets, setPresets] = useState<Preset[]>([])
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [appliedPreset, setAppliedPreset] = useState<string | null>(null)
  const [datasetId, setDatasetId] = useState('')
  const [devices, setDevices] = useState<number[]>([])
  const [advanced, setAdvanced] = useState(false)
  const [filter, setFilter] = useState('')
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [runName, setRunName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [candidates, setCandidates] = useState<WeightCandidate[]>([])
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [showReview, setShowReview] = useState(false)

  // 검증 결과에 "무엇을 검증했는지"를 같이 들고 있는다. 아래 check 계산이 이 값에 기댄다.
  const [modelCheck, setModelCheck] = useState<{ value: string; ok: boolean; message: string } | null>(null)

  const confirm = useConfirm()
  const prompt = usePrompt()
  const toast = useToast()

  useEffect(() => {
    api.paramsSchema().then((r) => {
      setSchema(r.schema)
      setValues(Object.fromEntries(r.schema.fields.map((f) => [f.key, f.default])))
    })
    api.presets().then((r) => setPresets(r.presets)).catch(() => {})
    api.weightCandidates().then((r) => setCandidates(r.candidates)).catch(() => {})
    api.systemInfo().then(setInfo).catch(() => {})
  }, [])

  /*
   * 모델 경로는 입력이 멈춘 뒤 서버에 물어본다. 학습을 시작하고 나서야 틀린 걸 아는 상황을 막는다.
   *
   * 디바운스 타이머만 취소하면 이미 날아간 요청은 못 막는다. 이전 경로의 "정상" 응답이
   * 늦게 도착하면 지금 입력된 틀린 경로가 통과된 것처럼 보인다. 그래서 effect 마다
   * cancelled 로 응답 자체를 버린다 — 값만 비교해서 걸러내면 그 응답이 state 에 남아
   * 현재 값과 영영 어긋난 채로 "확인 중" 에 잠긴다(새 요청은 modelValue 가 안 바뀌어 안 뜬다).
   */
  const modelValue = String(values['model'] ?? '')
  useEffect(() => {
    if (!modelValue) {
      setModelCheck(null)
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      api
        .validateModel(modelValue)
        .then((r) => {
          if (!cancelled) setModelCheck({ value: modelValue, ok: r.ok, message: r.message })
        })
        .catch(() => {
          // 확인 자체를 못 했으면 통과시키지 않는다. null 로 두면 영원히 "확인 중" 이다.
          if (!cancelled) setModelCheck({ value: modelValue, ok: false, message: '경로를 확인하지 못했습니다.' })
        })
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [modelValue])

  const check = modelCheck && modelCheck.value === modelValue ? modelCheck : null
  const checking = modelValue !== '' && check === null

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
      if (field.scope === 'options') continue // ultralytics CLI 인자가 아니다
      const v = values[field.key]
      if (v === field.default || v === undefined || v === null || v === '') continue
      parts.push(`${field.key}=${typeof v === 'string' && v.includes(' ') ? `"${v}"` : v}`)
    }
    if (values['model']) parts.push(`model="${values['model']}"`)
    parts.push(`device=${devices.length ? devices.join(',') : 'cpu'}`)
    return parts.join(' ')
  }, [schema, values, dataset, devices])

  function update(key: string, v: unknown) {
    setValues((s) => ({ ...s, [key]: v }))
    setAppliedPreset(null) // 손으로 고친 순간 더 이상 그 프리셋이 아니다
  }

  function scopedValues() {
    const params: Record<string, unknown> = {}
    const options: Record<string, unknown> = {}
    for (const f of schema?.fields ?? []) {
      ;(f.scope === 'options' ? options : params)[f.key] = values[f.key]
    }
    return { params, options }
  }

  async function start() {
    if (!dataset || !schema) return
    setBusy(true)
    setError('')
    try {
      // 서버는 params(학습 인자)와 options(UI 전용)를 각각 allowlist 로 검증한다.
      // 여기서 스코프별로 나눠 보내지 않으면 422 로 거절된다.
      const { params, options } = scopedValues()
      const run = await api.createRun({
        dataset_id: dataset.id,
        name: runName || dataset.name,
        devices,
        params,
        options,
      })
      onStarted(run.id)
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  async function savePreset() {
    if (!schema) return
    const name = await prompt({
      title: '프리셋 저장',
      label: '프리셋 이름',
      body: '지금 화면의 파라미터 전체를 이 이름으로 저장합니다.',
      confirmLabel: '저장',
    })
    if (!name) return
    const { params, options } = scopedValues()
    try {
      await api.savePreset(name, params, options)
      setPresets((await api.presets()).presets)
      setAppliedPreset(name)
      toast(`'${name}' 프리셋을 저장했습니다`, 'success')
    } catch (e) {
      toast(String(e instanceof Error ? e.message : e), 'error')
    }
  }

  async function deletePreset(preset: Preset) {
    const ok = await confirm({
      title: `'${preset.name}' 프리셋을 삭제할까요?`,
      confirmLabel: '삭제',
      danger: true,
    })
    if (!ok) return
    try {
      await api.deletePreset(preset.name)
      setPresets((await api.presets()).presets)
      if (appliedPreset === preset.name) setAppliedPreset(null)
    } catch (e) {
      toast(String(e instanceof Error ? e.message : e), 'error')
    }
  }

  // 서버 응답을 못 받은 상태를 "없음" 으로 접어 두지 않는다. 각각 다른 이유이고 다른 행동을 부른다.
  const blockedReason =
    datasetsStatus === 'error'
      ? '데이터셋 목록을 불러오지 못했습니다.'
      : gpusStatus === 'error'
        ? 'GPU 상태를 확인하지 못했습니다. 확인 전에는 시작하지 않습니다.'
        : !datasets.length
          ? datasetsStatus === 'loading'
            ? '데이터셋을 불러오는 중입니다.'
            : '먼저 데이터셋을 등록하세요.'
          : !dataset
            ? '학습할 데이터셋을 고르세요.'
            : check?.ok === false
              ? '모델 경로가 올바르지 않습니다.'
              : checking
                ? '모델 경로를 확인하는 중입니다.'
                : null

  return (
    <div className="pane">
      <DatasetRegister onDone={onDatasetsChanged} />

      <div className="card">
        <h3>학습 대상</h3>
        <div className="grid">
          <Field
            label="데이터셋"
            help={dataset ? <span className="mono">{dataset.yaml_path}</span> : undefined}
            status={
              datasetsStatus === 'error'
                ? { kind: 'bad', text: '목록을 불러오지 못했습니다.' }
                : undefined
            }
          >
            {(props) => (
              <select {...props} value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                {datasets.length === 0 && (
                  <option value="">
                    {datasetsStatus === 'loading'
                      ? '불러오는 중…'
                      : datasetsStatus === 'error'
                        ? '불러오지 못했습니다'
                        : '등록된 데이터셋이 없습니다'}
                  </option>
                )}
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} · {d.report.total_images}장 · {d.classes.length}클래스
                  </option>
                ))}
              </select>
            )}
          </Field>
          <Field label="실행 이름">
            {(props) => (
              <input
                {...props}
                value={runName}
                placeholder={dataset?.name ?? ''}
                onChange={(e) => setRunName(e.target.value)}
              />
            )}
          </Field>
        </div>

        <fieldset className="group" style={{ marginTop: 10, marginBottom: 0 }}>
          <legend className="small muted" style={{ marginBottom: 4 }}>
            사용할 GPU
          </legend>
          {/*
            "GPU 가 없다" 와 "GPU 목록을 못 받았다" 를 같은 문구로 보여주면, 서버가 잠깐 죽은 사이에
            사용자가 CPU 학습을 시작해 버린다. 실패는 실패라고 말하고 다시 시도할 길을 준다.
          */}
          {gpusStatus === 'error' ? (
            <div className="row tight small error">
              <span>GPU 상태를 확인하지 못했습니다.</span>
              <button className="btn-xs" onClick={onRetryGpus}>
                다시 시도
              </button>
            </div>
          ) : gpus.length === 0 ? (
            <div className="small muted">
              {gpusStatus === 'loading' ? 'GPU를 확인하는 중…' : 'GPU를 찾지 못했습니다. CPU로 학습합니다(매우 느립니다).'}
            </div>
          ) : (
            <div className="row wrap" style={{ gap: 12 }}>
              {gpus.map((g) => (
                <label key={g.index} className="row tight small">
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
        </fieldset>

        {dataset && (
          <button
            className="btn-sm"
            style={{ marginTop: 10 }}
            aria-expanded={showReview}
            onClick={() => setShowReview((s) => !s)}
          >
            {showReview ? '데이터셋 검수 접기' : '데이터셋 검수 보기'}
          </button>
        )}
      </div>

      {/* 파라미터를 고르기 전에 봐야 하는 정보다 — 작은 객체가 대부분이면 imgsz 를 키워야 한다. */}
      {dataset && showReview && <DatasetReviewPanel dataset={dataset} />}

      {dataset && schema && (
        <Recommendations
          dataset={dataset}
          /* 스코프를 나눠 보내야 한다. values 에는 UI 전용 옵션(tensorboard 등)이 섞여 있고,
             서버의 allowlist 는 그걸 params 로 받으면 422 로 거절한다 (start() 와 같은 이유). */
          values={scopedValues().params}
          devices={devices}
          onApply={(patch) => {
            setValues((s) => ({ ...s, ...patch }))
            setAppliedPreset(null) // 프리셋 위에 덮어썼으므로 더 이상 그 프리셋이 아니다
          }}
        />
      )}

      <div className="card">
        <div className="card-head">
          <h3>파라미터</h3>
          <label className="sr-only" htmlFor="param-filter">
            파라미터 이름으로 거르기
          </label>
          <input
            id="param-filter"
            type="search"
            className="small spacer"
            style={{ width: 160 }}
            placeholder="이름으로 필터"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <label className="row tight small muted nowrap">
            <input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)} />
            고급
          </label>
        </div>

        <div className="row wrap tight" role="group" aria-label="프리셋" style={{ marginBottom: 12 }}>
          <span className="small muted">프리셋</span>
          {presets.map((p) => (
            <span key={p.name} className="row" style={{ gap: 2 }}>
              <button
                className="btn-sm"
                aria-pressed={appliedPreset === p.name}
                title={p.builtin ? '내장 프리셋' : '내가 저장한 프리셋'}
                onClick={() => {
                  setValues((v) => ({ ...v, ...p.params, ...p.options }))
                  setAppliedPreset(p.name)
                }}
              >
                {p.builtin ? p.name : `★ ${p.name}`}
              </button>
              {!p.builtin && (
                <button className="btn-xs ghost" aria-label={`${p.name} 프리셋 삭제`} onClick={() => deletePreset(p)}>
                  ✕
                </button>
              )}
            </span>
          ))}
          <button className="btn-sm" onClick={savePreset}>
            + 현재 설정 저장
          </button>
        </div>

        {groups.map(({ group, fields }) => {
          const open = !collapsed.has(group)
          return (
            <fieldset className="group" key={group}>
              <legend>
                <button
                  type="button"
                  className="group-toggle"
                  aria-expanded={open}
                  onClick={() =>
                    setCollapsed((s) => {
                      const next = new Set(s)
                      if (open) next.add(group)
                      else next.delete(group)
                      return next
                    })
                  }
                >
                  <span className="chevron" aria-hidden="true">
                    ▾
                  </span>
                  {group}
                  <span className="tiny">{fields.length}개</span>
                </button>
              </legend>
              {open && (
                <div className="grid">
                  {fields.map((f) => (
                    <ParamFieldControl
                      key={f.key}
                      field={f}
                      value={values[f.key]}
                      onChange={(v) => update(f.key, v)}
                      candidates={f.key === 'model' ? candidates : undefined}
                      disabled={f.key === 'tensorboard' && info != null && !info.tensorboard}
                      status={
                        f.key === 'model' && check
                          ? { kind: check.ok ? 'ok' : 'bad', text: `${check.ok ? '✓' : '✕'} ${check.message}` }
                          : f.key === 'model' && checking
                            ? { kind: 'ok', text: '확인 중…' }
                            : undefined
                      }
                      help={
                        f.key === 'tensorboard' && info != null && !info.tensorboard
                          ? 'tensorboard 패키지가 설치되어 있지 않습니다.'
                          : undefined
                      }
                    />
                  ))}
                </div>
              )}
            </fieldset>
          )
        })}
      </div>

      <div className="card">
        <h3>동등한 CLI 명령</h3>
        <div className="log mono" style={{ maxHeight: 90, flex: 'none' }}>
          {cli}
        </div>
        <button
          className="btn-sm"
          style={{ marginTop: 8 }}
          disabled={!cli}
          onClick={() => {
            navigator.clipboard
              ?.writeText(cli)
              .then(() => toast('CLI 명령을 복사했습니다', 'success'))
              .catch(() => toast('클립보드에 복사하지 못했습니다', 'error'))
          }}
        >
          복사
        </button>
      </div>

      {error && <div className="card error">{error}</div>}

      <div className="sticky-foot">
        <div className="stack" style={{ flex: 1, gap: 4 }}>
          <button
            className="primary"
            style={{ padding: 10 }}
            disabled={busy || blockedReason !== null}
            onClick={start}
          >
            {busy ? '시작하는 중…' : '학습 시작'}
          </button>
          {blockedReason && (
            <span className="row tight small muted">
              <span>{blockedReason}</span>
              {datasetsStatus === 'error' && (
                <button className="btn-xs" onClick={onRetryDatasets}>
                  다시 시도
                </button>
              )}
              {gpusStatus === 'error' && (
                <button className="btn-xs" onClick={onRetryGpus}>
                  다시 시도
                </button>
              )}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * 스키마 한 필드를 실제 컨트롤로 바꾼다.
 *
 * 라벨 연결과 도움말 배치는 ui/Field 가 하고, "이 타입은 어떤 컨트롤인가" 같은
 * 도메인 판단만 여기 남긴다. 둘을 한 파일에 섞으면 Field 를 다른 화면에서 못 쓴다.
 */
function ParamFieldControl({
  field,
  value,
  onChange,
  candidates,
  disabled,
  status,
  help,
}: {
  field: ParamField
  value: unknown
  onChange: (v: unknown) => void
  candidates?: WeightCandidate[]
  disabled?: boolean
  status?: FieldStatus
  help?: ReactNode
}) {
  return (
    <Field
      label={field.label}
      labelExtra={<span className="mono muted tiny"> {field.key}</span>}
      help={help ?? field.help}
      status={status}
    >
      {(props) => {
        if (field.type === 'bool') {
          return <input {...props} type="checkbox" checked={!!value} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
        }
        if (field.type === 'path') {
          // 숫자 입력으로 렌더링되면 경로가 NaN 이 된다. 반드시 텍스트로 받는다.
          const listId = `${props.id}-list`
          return (
            <>
              <input
                {...props}
                type="text"
                spellCheck={false}
                list={listId}
                value={value === null || value === undefined ? '' : String(value)}
                placeholder="경로를 입력하거나 목록에서 고르세요"
                onChange={(e) => onChange(e.target.value)}
              />
              <datalist id={listId}>
                {(candidates ?? []).map((c) => (
                  <option key={c.value} value={c.value} label={`${c.label} — ${c.detail}`} />
                ))}
              </datalist>
            </>
          )
        }
        if (field.type === 'enum' && field.choices) {
          return (
            <select {...props} value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>
              {field.choices.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
          )
        }
        return (
          <input
            {...props}
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
        )
      }}
    </Field>
  )
}
