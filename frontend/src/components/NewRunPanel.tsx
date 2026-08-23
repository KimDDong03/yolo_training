import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api'
import { formatDuration } from '../format'
import type { Dataset, Estimate, Gpu, ParamField, ParamSchema, Preset, SystemInfo, WeightCandidate } from '../types'
import type { LoadStatus } from '../useResource'
import { DatasetReviewPanel } from './DatasetReviewPanel'
import { QualityPanel } from './QualityPanel'
import { useConfirm, usePrompt } from './ui/Dialog'
import { Recommendations, useAdvice } from './Recommendations'
import { TunePanel } from './TunePanel'
import { Field, type FieldStatus } from './ui/Field'
import { useToast } from './ui/Toast'

type GoalKey = 'quick' | 'balanced' | 'best'
type Mode = 'simple' | 'expert'

const MODE_KEY = 'yolo.newrun.mode'

/**
 * 목표 세 장.
 *
 * 균형의 imgsz 만 값이 비어 있다 — 그 자리는 데이터셋 검수에서 나온 서버 추천이 채운다.
 * 시안의 960 은 그 데이터셋에서 서버가 낸 값이지 상수가 아니다. 추천이 없으면 스키마
 * 기본값을 쓰고 강조하지 않는다(강조는 "검수 때문에 바뀐 값" 이라는 뜻이라서).
 */
/**
 * 한 번에 훑을 수 있는 값의 개수.
 *
 * 백엔드의 runs.py SWEEP_MAX 와 Sidebar 의 COMPARE_LIMIT 과 **같은 값이어야 한다** -
 * 스윕 결과는 비교 화면에 통째로 들어가야 의미가 있다. 셋 중 하나만 바꾸면 어긋난다.
 */
const SWEEP_MAX = 6

const GOALS: { key: GoalKey; title: string; tag?: string; desc: string; patch: Record<string, number> }[] = [
  { key: 'quick', title: '빠른 확인', desc: '파이프라인이 도는지 먼저 본다', patch: { epochs: 3, imgsz: 320, batch: 8 } },
  // imgsz 0 은 자리표시자다. 아래 goalPatch 가 추천값으로 바꿔 넣는다 — 키 순서를 지키려고 여기 둔다.
  { key: 'balanced', title: '균형', tag: '추천', desc: '이 데이터셋 크기에 맞춘 기본값', patch: { epochs: 100, imgsz: 0, batch: 16 } },
  { key: 'best', title: '최고 정확도', desc: '시간을 더 쓰고 mAP를 끌어올린다', patch: { epochs: 300, imgsz: 1280, batch: 8 } },
]

/** 값이 바뀐 경로. 목표 카드와 프리셋 배지가 언제 풀리는지를 이 한 곳에서 정한다. */
type Source =
  | { kind: 'goal'; goal: GoalKey }
  | { kind: 'preset'; name: string }
  | { kind: 'manual' }
  | { kind: 'recommend' }
  | { kind: 'tune' }

interface Props {
  datasets: Dataset[]
  datasetsStatus: LoadStatus
  onRetryDatasets: () => void
  gpus: Gpu[]
  gpusStatus: LoadStatus
  onRetryGpus: () => void
  onRegisterDataset: () => void
  onStarted: (runId: string) => void
  /** 스윕이 만든 run 들. 곧바로 비교 화면으로 보낸다. */
  onSweepStarted: (runIds: string[]) => void
}

export function NewRunPanel({
  datasets,
  datasetsStatus,
  onRetryDatasets,
  gpus,
  gpusStatus,
  onRetryGpus,
  onRegisterDataset,
  onStarted,
  onSweepStarted,
}: Props) {
  const [schema, setSchema] = useState<ParamSchema | null>(null)
  const [presets, setPresets] = useState<Preset[]>([])
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [appliedPreset, setAppliedPreset] = useState<string | null>(null)
  const [goal, setGoal] = useState<GoalKey | null>(null)
  const [mode, setMode] = useState<Mode>(() =>
    (localStorage.getItem(MODE_KEY) as Mode | null) === 'expert' ? 'expert' : 'simple',
  )
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

  useEffect(() => {
    localStorage.setItem(MODE_KEY, mode)
  }, [mode])

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

  function scopedValues(source: Record<string, unknown> = values) {
    const params: Record<string, unknown> = {}
    const options: Record<string, unknown> = {}
    for (const f of schema?.fields ?? []) {
      ;(f.scope === 'options' ? options : params)[f.key] = source[f.key]
    }
    return { params, options }
  }

  const { rec, est } = useAdvice(dataset, scopedValues().params, devices)

  // 검수에서 나온 imgsz 제안. 균형 카드가 이 값을 쓰고, 같은 색으로 강조한다.
  const recImgsz = typeof rec?.patch?.imgsz === 'number' ? rec.patch.imgsz : null
  const defaultImgsz = schema?.fields.find((f) => f.key === 'imgsz')?.default
  const goalPatch = useMemo(
    () =>
      Object.fromEntries(
        GOALS.map((g) => [
          g.key,
          g.key === 'balanced' ? { ...g.patch, imgsz: recImgsz ?? Number(defaultImgsz ?? 640) } : g.patch,
        ]),
      ) as Record<GoalKey, Record<string, number>>,
    [recImgsz, defaultImgsz],
  )

  const goalTimes = useGoalTimes(dataset, devices, values, goalPatch, schema)
  // 간편 모드에도 시간 예산을 내보낸다. 파라미터 폼 전체가 전문 모드 안에만 있어서
  // 이대로 두면 이 기능이 겨냥한 사용자가 볼 수 없다. 정의는 스키마에서 읽는다.
  const timeField = schema?.fields.find((f) => f.key === 'time')

  const [sweepOn, setSweepOn] = useState(false)
  const [sweepAxis, setSweepAxis] = useState('imgsz')
  const [sweepText, setSweepText] = useState('')

  const budgetOn = Number(values['time'] ?? 0) > 0
  /*
   * 훑을 수 있는 축. bool 은 값이 둘뿐이라 스윕이랄 게 없고, 예산이 켜져 있으면
   * epochs 는 ultralytics 가 덮어써서(trainer.py:546) 모든 실행이 같아진다.
   * 서버도 같은 규칙으로 거절하지만(POST /api/runs/sweep) 고를 수 없게 먼저 막는다.
   */
  const sweepAxes = useMemo(
    () =>
      (schema?.fields ?? []).filter(
        (f) =>
          f.scope === 'params' &&
          f.type !== 'bool' &&
          !(f.key === 'epochs' && budgetOn),
      ),
    [schema, budgetOn],
  )
  const axisField = sweepAxes.find((f) => f.key === sweepAxis) ?? sweepAxes[0]
  const sweepValues = useMemo(
    () => parseSweepValues(sweepText, axisField),
    [sweepText, axisField],
  )
  const sweepTimes = useSweepTimes(
    dataset, devices, values, schema, sweepOn ? axisField : undefined, sweepValues.ok,
  )

  /*
   * 추천은 늦게 온다. 그 전에 균형을 고르면 카드에는 추천 imgsz 가 뜨는데 폼에는 기본값이
   * 남아, 고른 카드와 실제 설정이 어긋난다. 추천이 확정되면 고른 카드를 다시 적용한다.
   * patch 가 그대로면 setValues 가 같은 값을 쓰므로 반복되지 않는다.
   */
  useEffect(() => {
    if (!goal) return
    const patch = goalPatch[goal]
    setValues((v) => (Object.entries(patch).every(([k, x]) => v[k] === x) ? v : { ...v, ...patch }))
  }, [goal, goalPatch])

  /**
   * 값을 바꾸는 유일한 경로.
   *
   * 예전에는 수동 수정만 update() 를 거치고 프리셋·추천·탐색은 setValues 를 직접 불렀다.
   * 그래서 "손대면 풀린다" 규칙이 절반만 걸렸고, 카드가 선택된 채 값만 달라질 수 있었다.
   */
  function applyValues(patch: Record<string, unknown>, source: Source) {
    setValues((s) => ({ ...s, ...patch }))
    setGoal(source.kind === 'goal' ? source.goal : null)
    setAppliedPreset(source.kind === 'preset' ? source.name : null)
  }

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

  // 숫자를 하드코딩하지 않는다. 정본은 47개라고 적었지만 실제 스키마는 46개다.
  const fieldCount = schema?.fields.length ?? 0
  const changedCount = useMemo(
    () => (schema?.fields ?? []).filter((f) => values[f.key] !== f.default).length,
    [schema, values],
  )

  async function startSweep() {
    if (!dataset || !schema || !axisField) return
    setBusy(true)
    setError('')
    try {
      const { params, options } = scopedValues()
      const out = await api.createSweep({
        dataset_id: dataset.id,
        name: runName || dataset.name,
        devices,
        params,
        options,
        axis: axisField.key,
        values: sweepValues.ok,
      })
      onSweepStarted(out.runs.map((r) => r.id))
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
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

  const issues = Object.values(dataset?.report.issue_counts ?? {}).reduce((a, b) => a + b, 0)

  return (
    <div className="new-run">
      <div className="new-run-scroll">
        <div className="new-run-body">
          <div className="new-run-head">
            <h2>새 학습</h2>
            <span className="muted">세 가지만 정하면 됩니다</span>
            <div className="segmented spacer" role="radiogroup" aria-label="설정 모드">
              {(['simple', 'expert'] as Mode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  role="radio"
                  aria-checked={mode === m}
                  className="tiny"
                  onClick={() => setMode(m)}
                >
                  {m === 'simple' ? '간편' : '전문'}
                </button>
              ))}
            </div>
          </div>

          <section className="section">
            <div className="section-label">01</div>
            <h3>무엇으로 학습할까요</h3>
            <div className="card">
              <div className="row">
                <select
                  aria-label="데이터셋"
                  value={datasetId}
                  onChange={(e) => setDatasetId(e.target.value)}
                  style={{ flex: 1 }}
                >
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
                      {d.name}
                    </option>
                  ))}
                </select>
                <button className="ghost nowrap" onClick={onRegisterDataset}>
                  ＋ 데이터셋 등록
                </button>
              </div>

              {dataset && (
                <div className="summary-line" style={{ marginTop: 'var(--sp-3)' }}>
                  <span>이미지 {dataset.report.total_images.toLocaleString()}</span>
                  <span>클래스 {dataset.classes.length}</span>
                  {dataset.report.train_count != null && dataset.report.val_count != null && (
                    <span>
                      train/val {dataset.report.train_count.toLocaleString()} /{' '}
                      {dataset.report.val_count.toLocaleString()}
                    </span>
                  )}
                  {issues > 0 && <span className="warn spacer">검수 경고 {issues}건</span>}
                  <button
                    className="btn-xs ghost"
                    style={{ marginLeft: issues > 0 ? 0 : 'auto' }}
                    aria-expanded={showReview}
                    onClick={() => setShowReview((s) => !s)}
                  >
                    {showReview ? '리포트 접기' : '리포트 →'}
                  </button>
                </div>
              )}

              <Advice rec={rec} />
            </div>
          </section>

          {/* 파라미터를 고르기 전에 봐야 하는 정보다 — 작은 객체가 대부분이면 imgsz 를 키워야 한다. */}
          {dataset && showReview && (
            <>
              <DatasetReviewPanel dataset={dataset} />
              {/* 누수는 학습을 시작하기 전에 알아야 한다. 돌린 뒤에 알면 그 mAP 를 버리게 된다. */}
              <QualityPanel dataset={dataset} />
            </>
          )}

          <section className="section">
            <div className="section-label">02</div>
            <h3>어디까지 돌릴까요</h3>
            <div className="goal-grid" role="group" aria-label="학습 목표">
              {GOALS.map((g) => (
                <button
                  key={g.key}
                  className="goal"
                  aria-pressed={goal === g.key}
                  onClick={() => applyValues(goalPatch[g.key], { kind: 'goal', goal: g.key })}
                >
                  <span className="row tight">
                    <span className="goal-radio" aria-hidden="true" />
                    <span className="goal-title">{g.title}</span>
                    {g.tag && <span className="badge">{g.tag}</span>}
                  </span>
                  <span className="goal-desc">{g.desc}</span>
                  <span className="goal-params">
                    {Object.entries(goalPatch[g.key]).map(([k, v]) => (
                      <span key={k} style={{ display: 'block' }}>
                        {k}{' '}
                        <span className={g.key === 'balanced' && k === 'imgsz' && recImgsz ? 'from-review' : undefined}>
                          {v}
                        </span>
                      </span>
                    ))}
                  </span>
                  <span className="goal-time">
                    {goalTimes[g.key] == null ? '예상 시간 계산 중…' : <>약 <strong>{formatDuration(goalTimes[g.key]!)}</strong></>}
                  </span>
                </button>
              ))}
            </div>

            {/*
              에폭이 아니라 시간으로 끊고 싶을 때. 카드와 배타가 아니라 위에 얹는 것이라
              카드의 patch 는 그대로 두고 여기서 값만 더한다 - 0 이면 꺼진 상태다.
            */}
            {timeField && (
              <div className="budget-row">
                <ParamFieldControl
                  field={timeField}
                  value={values[timeField.key]}
                  changed={values[timeField.key] !== timeField.default}
                  onChange={(v) => applyValues({ [timeField.key]: v }, { kind: 'manual' })}
                />
              </div>
            )}

            {/*
              한 축을 여러 값으로 훑는다. 축 정의는 스키마에서 읽고(param_schema.py:3 방침)
              만들어질 run 을 먼저 보여 준다 - 밤새 도는 일이라 누르기 전에 총 시간을 봐야 한다.
            */}
            {schema && axisField && (
              <div className="budget-row sweep-row">
                <label className="row tight">
                  <input type="checkbox" checked={sweepOn} onChange={(e) => setSweepOn(e.target.checked)} />
                  <span>한 축을 여러 값으로 훑기</span>
                </label>

                {sweepOn && (
                  <div className="sweep-body">
                    <div className="row tight">
                      <select
                        aria-label="훑을 항목"
                        value={axisField.key}
                        onChange={(e) => setSweepAxis(e.target.value)}
                      >
                        {sweepAxes.map((f) => (
                          <option key={f.key} value={f.key}>
                            {f.label} ({f.key})
                          </option>
                        ))}
                      </select>
                      <input
                        type="text"
                        aria-label="훑을 값"
                        className="spacer"
                        spellCheck={false}
                        placeholder={axisField.type === 'enum' ? 'SGD, AdamW' : '320, 416, 512'}
                        value={sweepText}
                        onChange={(e) => setSweepText(e.target.value)}
                      />
                    </div>

                    {sweepValues.error && <p className="small error">{sweepValues.error}</p>}

                    {sweepValues.ok.length > 0 && (
                      <ul className="sweep-preview">
                        {sweepValues.ok.map((v, i) => (
                          <li key={String(v)}>
                            <span className="chip" aria-hidden="true">{String(v)}</span>
                            <span className="mono small muted">
                              {(runName || dataset?.name || '')}/{axisField.key}={String(v)}
                            </span>
                            <span className="small muted spacer nowrap">
                              {sweepTimes[i] == null ? '계산 중…' : `약 ${formatDuration(sweepTimes[i]!)}`}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}

                    {sweepValues.ok.length > 0 && (
                      <p className="small muted">
                        {sweepTimes.every((t) => t != null)
                          ? `전부 합쳐 약 ${formatDuration(sweepTimes.reduce((a, b) => a + (b ?? 0), 0))}`
                          : '합계를 계산하는 중입니다'}
                        {!budgetOn && (
                          <>
                            {' · '}
                            <strong>시간 예산이 꺼져 있어 값마다 소요가 다릅니다.</strong> 같은 비용에서
                            비교하려면 위에서 예산을 켜세요.
                          </>
                        )}
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="section">
            <div className="section-label">03</div>
            <h3>어디서 돌릴까요</h3>
            <div className="card">
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
                  {gpusStatus === 'loading'
                    ? 'GPU를 확인하는 중…'
                    : 'GPU를 찾지 못했습니다. CPU로 학습합니다(매우 느립니다).'}
                </div>
              ) : (
                <div className="gpu-grid">
                  {gpus.map((g) => {
                    const on = devices.includes(g.index)
                    const freeGb = Math.round((g.memory_total_mb - g.memory_used_mb) / 1024)
                    return (
                      <label key={g.index} className={`gpu-pick${on ? ' on' : ''}`}>
                        <input
                          type="checkbox"
                          checked={on}
                          onChange={(e) =>
                            setDevices((d) =>
                              e.target.checked ? [...d, g.index].sort((a, b) => a - b) : d.filter((x) => x !== g.index),
                            )
                          }
                        />
                        <span style={{ flex: 1, minWidth: 0 }}>
                          GPU #{g.index} · {g.name}
                        </span>
                        <span className="meter" aria-hidden="true">
                          <div
                            style={{ width: `${Math.min(100, (1 - g.memory_used_mb / g.memory_total_mb) * 100)}%` }}
                          />
                        </span>
                        <span className="gpu-free">{freeGb}GB 여유</span>
                      </label>
                    )
                  })}
                </div>
              )}
              {devices.length > 1 && (
                <div className="help">GPU를 2장 이상 고르면 ultralytics가 DDP(분산 학습)로 실행합니다.</div>
              )}

              <div style={{ marginTop: 'var(--sp-4)' }}>
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
            </div>
          </section>

          {mode === 'expert' && dataset && schema && (
            <>
              <Recommendations
                values={scopedValues().params}
                rec={rec}
                est={est}
                onApply={(patch) => applyValues(patch, { kind: 'recommend' })}
              />

              {/* 규칙 추천이 데이터셋 통계로 정하는 값 위에, 실제로 돌려 본 값을 얹는다.
                  결과는 같은 경로로 폼에 들어간다. */}
              <TunePanel
                dataset={dataset}
                model={String(values.model ?? '')}
                gpuCount={gpus.length}
                onApply={(patch) => applyValues(patch, { kind: 'tune' })}
              />
            </>
          )}

          {mode === 'expert' ? (
            <>
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
                    고급 항목 포함
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
                        onClick={() => applyValues({ ...p.params, ...p.options }, { kind: 'preset', name: p.name })}
                      >
                        {p.builtin ? p.name : `★ ${p.name}`}
                      </button>
                      {!p.builtin && (
                        <button
                          className="btn-xs ghost"
                          aria-label={`${p.name} 프리셋 삭제`}
                          onClick={() => deletePreset(p)}
                        >
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
                              changed={values[f.key] !== f.default}
                              onChange={(v) => applyValues({ [f.key]: v }, { kind: 'manual' })}
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
              </div>
            </>
          ) : (
            <button className="more-params" onClick={() => setMode('expert')}>
              <span aria-hidden="true">▸</span>
              <span>세부 파라미터 {fieldCount}개</span>
              {changedCount > 0 && <span className="changed">기본값에서 {changedCount}개 변경됨</span>}
            </button>
          )}

          {error && <div className="card error">{error}</div>}
        </div>
      </div>

      <div className="new-run-foot">
        <div style={{ flex: 1, minWidth: 0 }}>
          <span className="foot-summary">
            {[
              runName || dataset?.name || '이름 없음',
              `${values['epochs'] ?? '-'}에폭`,
              `${values['imgsz'] ?? '-'}px`,
              devices.length ? `GPU #${devices.join(', #')}` : 'CPU',
            ].join(' · ')}
          </span>
          <span className="foot-sub">
            {blockedReason ? (
              <span className="row tight">
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
            ) : est?.ok ? (
              `예상 ${formatDuration(est.total_time_s)} · 종료 예정 ${new Date(Date.now() + est.total_time_s * 1000).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false })}`
            ) : (
              '예상 시간을 계산하는 중입니다'
            )}
          </span>
        </div>

        <button
          className="nowrap"
          disabled={!cli}
          onClick={() => {
            navigator.clipboard
              ?.writeText(cli)
              .then(() => toast('CLI 명령을 복사했습니다', 'success'))
              .catch(() => toast('클립보드에 복사하지 못했습니다', 'error'))
          }}
        >
          CLI 명령 복사
        </button>
        <button
          className="primary nowrap"
          style={{ fontSize: 15, padding: '0 36px' }}
          disabled={busy || blockedReason !== null || (sweepOn && sweepValues.ok.length < 2)}
          onClick={sweepOn ? startSweep : start}
        >
          {busy
            ? '시작하는 중…'
            : sweepOn
              ? `스윕 시작 (${sweepValues.ok.length}개)`
              : '학습 시작'}
        </button>
      </div>
    </div>
  )
}

/**
 * 쉼표로 구분한 입력을 축 타입에 맞는 값으로 바꾼다.
 *
 * 여기 검증은 UX 다 - 진짜 관문은 서버의 param_schema.validate 다(param_schema.py:250).
 * 그래도 min/max 를 먼저 보는 이유는, 값 하나가 범위를 벗어나면 서버가 스윕 전체를
 * 422 로 거절하기 때문이다. 누르고 나서 알면 늦다.
 */
function parseSweepValues(
  text: string,
  field: ParamField | undefined,
): { ok: unknown[]; error: string } {
  if (!field) return { ok: [], error: '' }
  const parts = text.split(',').map((t) => t.trim()).filter(Boolean)
  if (parts.length === 0) return { ok: [], error: '' }

  const ok: unknown[] = []
  for (const part of parts) {
    if (field.type === 'int' || field.type === 'float') {
      const n = field.type === 'int' ? Number.parseInt(part, 10) : Number.parseFloat(part)
      if (!Number.isFinite(n)) return { ok: [], error: `숫자로 읽을 수 없습니다: ${part}` }
      if (field.min != null && n < field.min)
        return { ok: [], error: `${part} 은 최소값 ${field.min} 보다 작습니다.` }
      if (field.max != null && n > field.max)
        return { ok: [], error: `${part} 은 최대값 ${field.max} 보다 큽니다.` }
      ok.push(n)
    } else if (field.type === 'enum') {
      const allowed = (field.choices ?? []).map((c) => c.value)
      if (allowed.length && !allowed.includes(part))
        return { ok: [], error: `허용되지 않은 값입니다: ${part} (${allowed.join(', ')})` }
      ok.push(part)
    } else {
      ok.push(part)
    }
  }
  if (new Set(ok.map(String)).size !== ok.length)
    return { ok: [], error: '같은 값을 두 번 넣을 수 없습니다.' }
  if (ok.length > SWEEP_MAX)
    return { ok: [], error: `한 번에 ${SWEEP_MAX}개까지만 훑을 수 있습니다.` }
  return { ok, error: '' }
}

/**
 * 값마다 예상 시간을 받아 온다. useGoalTimes 와 같은 구조다 - 값이 바뀔 때만 돌게
 * 서명을 만들고 300ms 미룬다.
 */
function useSweepTimes(
  dataset: Dataset | undefined,
  devices: number[],
  values: Record<string, unknown>,
  schema: ParamSchema | null,
  axis: ParamField | undefined,
  sweepValues: unknown[],
): (number | null)[] {
  const [times, setTimes] = useState<(number | null)[]>([])

  const signature = JSON.stringify([
    dataset?.id,
    devices,
    axis?.key,
    sweepValues,
    // 축이 덮지 않으면서 estimate 에 영향을 주는 값들. useGoalTimes 와 같은 목록이다.
    values['model'],
    values['imgsz'],
    values['epochs'],
    values['batch'],
    values['amp'],
    values['patience'],
    values['mixup'],
    values['cache'],
    values['close_mosaic'],
    values['time'],
  ])

  useEffect(() => {
    if (!dataset || !schema || !axis || sweepValues.length === 0) {
      setTimes([])
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      Promise.all(
        sweepValues.map((v) => {
          const merged = { ...values, [axis.key]: v }
          const params: Record<string, unknown> = {}
          for (const f of schema.fields) {
            if (f.scope !== 'options') params[f.key] = merged[f.key]
          }
          return api
            .estimate(dataset.id, params, devices)
            .then((e: Estimate) => (e.ok ? e.total_time_s : null))
            .catch(() => null)
        }),
      ).then((results) => {
        if (!cancelled) setTimes(results)
      })
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature])

  return times
}

/**
 * 검수 결과를 결정 지점에 붙인다.
 *
 * 문장은 서버가 만든다(`recommend.py`). 여기서 다시 쓰면 진단 화면과 다른 말을 하게 된다.
 * 값이 아니라 알려만 주는 advisory 는 여기 넣지 않는다 — 그건 전문 모드의 표가 맡는다.
 */
function Advice({ rec }: { rec: { items: { reason: string; effect: string }[] } | null }) {
  // 정본은 이 문장의 숫자를 강조하지만, 문장이 서버에서 통째로 오므로 숫자를 골라내려면
  // 한국어 문장을 파싱해야 한다. 엉뚱한 숫자를 칠할 위험이 있어 하지 않는다 —
  // 검수와 카드를 잇는 강조는 목표 카드 쪽 .from-review 가 맡는다.
  const first = rec?.items?.[0]
  if (!first) return null
  return (
    <div className="advice">
      <strong>{first.reason}</strong> {first.effect}
    </div>
  )
}

/**
 * 목표 카드 세 장의 예상 시간.
 *
 * 카드마다 서버에 물어본다 — epoch 수만으로는 못 구한다. imgsz 가 바뀌면 에폭 하나에
 * 걸리는 시간 자체가 달라지기 때문이다. 비율로 곱해 지어내지 않는다.
 * /api/estimate 는 보정표에서 계산만 하므로 세 번 불러도 부담이 없고, 카드 값이 바뀔 때만 돈다.
 */
function useGoalTimes(
  dataset: Dataset | undefined,
  devices: number[],
  values: Record<string, unknown>,
  goalPatch: Record<GoalKey, Record<string, number>>,
  schema: ParamSchema | null,
): Record<GoalKey, number | null> {
  const [times, setTimes] = useState<Record<GoalKey, number | null>>({ quick: null, balanced: null, best: null })

  /*
   * goalPatch 가 덮어쓰지 않는 값 가운데 estimate 에 실제로 영향을 주는 것만 넣는다
   * (epochs·imgsz·batch 는 카드가 덮으므로 goalPatch 에 이미 들어 있다).
   * 이걸 빼면 전문 모드에서 amp 를 꺼도 카드의 예상 시간이 이전 값으로 남는다.
   */
  const signature = JSON.stringify([
    dataset?.id,
    devices,
    goalPatch,
    values['model'],
    values['amp'],
    values['patience'],
    values['mixup'],
    values['cache'],
    values['close_mosaic'],
    // 시간 예산은 카드가 덮지 않는데 estimate 결과를 통째로 바꾼다. 빼면 예산을 바꿔도
    // 카드의 예상 시간이 이전 값으로 남는다 - 위 amp 와 같은 함정이다.
    values['time'],
  ])

  useEffect(() => {
    if (!dataset || !schema) return
    let cancelled = false
    const timer = setTimeout(() => {
      Promise.all(
        GOALS.map((g) => {
          const merged = { ...values, ...goalPatch[g.key] }
          const params: Record<string, unknown> = {}
          for (const f of schema.fields) {
            if (f.scope !== 'options') params[f.key] = merged[f.key]
          }
          return api
            .estimate(dataset.id, params, devices)
            .then((e: Estimate) => (e.ok ? e.total_time_s : null))
            .catch(() => null)
        }),
      ).then((results) => {
        if (cancelled) return
        setTimes({ quick: results[0], balanced: results[1], best: results[2] })
      })
    }, 300)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, schema])

  return times
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
  changed,
}: {
  field: ParamField
  value: unknown
  onChange: (v: unknown) => void
  candidates?: WeightCandidate[]
  disabled?: boolean
  status?: FieldStatus
  help?: ReactNode
  /** 기본값에서 벗어난 필드. 46개를 훑을 때 무엇을 건드렸는지가 먼저 보여야 한다. */
  changed?: boolean
}) {
  return (
    <div className={changed ? 'field-changed' : undefined}>
    <Field
      label={field.label}
      labelExtra={
        <>
          <span className="mono muted tiny"> {field.key}</span>
          {changed && <span className="changed-tag">변경됨</span>}
        </>
      }
      help={help ?? field.help}
      status={status}
    >
      {(props) => {
        if (field.type === 'bool') {
          return (
            <input
              {...props}
              type="checkbox"
              checked={!!value}
              disabled={disabled}
              onChange={(e) => onChange(e.target.checked)}
            />
          )
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
    </div>
  )
}
