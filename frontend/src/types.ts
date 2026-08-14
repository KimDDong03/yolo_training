export type ParamType = 'int' | 'float' | 'bool' | 'enum' | 'str' | 'path'

export interface WeightCandidate {
  value: string
  label: string
  detail: string
  kind: 'weights' | 'config'
}

export interface ModelCheck {
  ok: boolean
  kind: 'weights' | 'config' | 'unknown'
  resolved: string | null
  message: string
}

export interface Preset {
  name: string
  params: Record<string, unknown>
  options: Record<string, unknown>
  created_at: number | null
  builtin: boolean
}

export interface ReviewCategory {
  code: string
  label: string
  total: number
  stored: number
  truncated: boolean
}

export interface BoxStats {
  count: number
  area: { label: string; count: number }[]
  aspect: { label: string; count: number }[]
  tiny_ratio: number
  median_area?: number
}

export interface DatasetReview {
  categories: ReviewCategory[]
  box_stats: BoxStats
  review_cap: number
  category: string
  page: {
    items: { path: string; detail: string }[]
    total: number
    stored: number
    truncated: boolean
  }
}

export interface ExportStatus {
  status: 'idle' | 'running' | 'completed' | 'failed'
  events: { t: string; [k: string]: unknown }[]
  result: { status?: string; file?: string; size_mb?: number; error?: string } | null
  format: string | null
}

export interface Detection {
  cls: number
  name: string
  conf: number
  xyxy: [number, number, number, number]
}

export interface PredictResult {
  image: string
  detections: Detection[]
  count: number
  elapsed_ms: number
  weights: string
  device: string
}

export interface SystemInfo {
  tensorboard: boolean
  tensorrt: boolean
  onnx: boolean
}

export interface Choice {
  value: string
  label: string
  available: boolean
}

export interface ParamField {
  key: string
  label: string
  type: ParamType
  group: string
  advanced: boolean
  default: unknown
  min: number | null
  max: number | null
  step: number | null
  choices: Choice[] | null
  help: string
  /** params = ultralytics 학습 인자, options = UI 전용 값. 서버가 이 기준으로 나눠 검증한다. */
  scope: 'params' | 'options'
}

export interface ParamSchema {
  groups: string[]
  fields: ParamField[]
}

export interface Gpu {
  index: number
  name: string
  memory_total_mb: number
  memory_used_mb: number
  utilization: number
}

export interface DatasetReport {
  total_images: number
  train_count?: number
  val_count?: number
  auto_split?: boolean
  val_ratio?: number | null
  split_counts: Record<string, number>
  class_instances?: Record<string, number>
  /** 카테고리 코드 → 건수. 목록 자체는 review.json 에 따로 있다(GET /api/datasets/{id}/review). */
  issue_counts: Record<string, number>
  box_stats?: BoxStats
  review_cap: number
}

export interface Dataset {
  id: string
  name: string
  source: 'zip' | 'path'
  origin: string
  root: string
  yaml_path: string
  classes: string[]
  report: DatasetReport
  created_at: number
}

export type RunStatus = 'queued' | 'running' | 'completed' | 'stopped' | 'failed'

export interface Run {
  id: string
  name: string
  dataset_id: string
  status: RunStatus
  params: Record<string, unknown>
  /** UI 전용 값. 이 컬럼이 생기기 전 run 은 빈 객체로 온다. */
  options: Record<string, unknown>
  devices: number[]
  pid: number | null
  error: string | null
  created_at: number
  started_at: number | null
  finished_at: number | null
  dataset?: Dataset | null
}

export interface TrainEvent {
  t: 'start' | 'batch' | 'epoch' | 'final_val' | 'artifact' | 'checkpoint' | 'end'
  ts: number
  epoch?: number
  total_epochs?: number
  metrics?: Record<string, number | null>
  summary?: Record<string, number | null>
  lr?: Record<string, number | null>
  epoch_time_s?: number
  eta_s?: number
  files?: string[]
  i?: number
  n?: number | null
  loss?: number | null
  status?: string
  error?: string
  plots?: string[]
  weights?: string[]
  classes?: string[]
  device?: string
  model?: string
}

export interface Artifacts {
  plots: string[]
  weights: string[]
  epochs: Record<string, string[]>
}

/**
 * 지금 화면에 무엇이 떠 있는가.
 *
 * 예전에는 `selected: string | null` 하나로 "run 상세" 와 "새 학습 설정" 을 겸했다.
 * 그래서 새 학습으로 돌아가려면 셀렉트에서 빈 항목을 다시 고르는 수밖에 없었다.
 * 화면을 명시적으로 나열해 두면 그런 숨은 모드 전환이 사라진다.
 */
export type View =
  | { kind: 'new' }
  | { kind: 'run'; id: string }
  | { kind: 'datasets' }
  | { kind: 'compare' }
