export type ParamType = 'int' | 'float' | 'bool' | 'enum' | 'str'

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
  missing_labels: string[]
  empty_labels: string[]
  orphan_labels: string[]
  label_issues: { file: string; issues: string[] }[]
  detailed_report: boolean
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
