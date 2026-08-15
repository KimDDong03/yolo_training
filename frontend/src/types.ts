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

/**
 * 등록된 원본 경로가 아직 쓸 수 있는가. 서버가 요청마다 계산해 얹는다.
 *
 * 경로 참조 데이터셋의 폴더를 옮기면 사진만 조용히 전부 깨진다(학습·분석은 목록 파일을
 * 쓰므로 계속 동작한다). 이 키가 없는 응답은 이 검사가 생기기 전의 서버다.
 */
export interface DatasetPathStatus {
  ok: boolean
  code: 'ok' | 'root_missing' | 'list_missing' | 'images_missing' | 'outside_root'
  message: string
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
  path_status?: DatasetPathStatus
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
  /** 이 실행이 재시도라면 원본 실행 ID. */
  retry_of: string | null
  created_at: number
  started_at: number | null
  finished_at: number | null
  dataset?: Dataset | null
}

/** 재시도할 때 바뀌는 값 하나. */
export interface DiagnosisChange {
  from: unknown
  to: unknown
}

/** 이 데이터셋에는 이 값이 낫다는 제안 하나. */
export interface RecommendationItem {
  rule: string
  severity: 'info' | 'warn'
  changes: Record<string, DiagnosisChange>
  /** 왜 이 제안이 나왔는지 (데이터에서 관측된 사실). */
  reason: string
  /** 적용하면 무엇이 좋아지고 무엇을 치르는지. */
  effect: string
}

export interface Recommendation {
  /** false 면 이 데이터셋에 박스 분포 정보가 없다는 뜻이다 (구버전 등록). */
  available: boolean
  reason?: string
  items: RecommendationItem[]
  /** 값은 건드리지 않고 알려만 주는 것들. */
  advisories: { code: string; severity: 'info' | 'warn'; message: string }[]
  /** 모든 제안을 합친 것. 한 번에 적용할 때 쓴다. */
  patch: Record<string, unknown>
}

/** 사이드잡(내보내기·분석) 공통 상태. */
export interface JobStatus {
  status: 'idle' | 'running' | 'completed' | 'failed' | 'stopped'
  kind?: string
  args?: Record<string, unknown>
  devices?: number[]
  events: { t: string; stage?: string; message?: string; status?: string; error?: string }[]
  result: Record<string, unknown> | null
  error?: string | null
}

export interface AnalysisBox {
  cls: number
  name: string
  /** [x1, y1, x2, y2], 각각 0~1. */
  box: [number, number, number, number]
  conf?: number | null
  /** 정답: hit=찾음 miss=놓침 / 예측: hit=맞음 false=오검출 */
  state: 'hit' | 'miss' | 'false'
}

export type TideKind = 'cls' | 'loc' | 'both' | 'dupe' | 'bkg' | 'miss'

export interface TideError {
  kind: TideKind
  label: string
  /** 점수 계산에 쓰인 전체 검출 기준 건수. conf 0.001 짜리 잡음이 대부분이다. */
  count: number
  /** 배포 임계값에서 실제로 보이는 건수. 놓침은 이 기준으로 셀 수 없어 null 이다. */
  count_at_conf: number | null
  /** 이 유형만 고쳤을 때의 mAP50 상승분. 무엇부터 손댈지가 이 값의 크기 순이다. */
  dap: number | null
  /** 분모 보정 전 값. 진단용이라 화면에는 쓰지 않는다. */
  dap_naive: number | null
  /** 놓침을 고치면서 정답이 하나도 안 남게 된 클래스. */
  dropped_classes: number[]
  /** 양수 상승분 합에서 이 유형이 차지하는 비율. 전부 0 이면 null. */
  share: number | null
  /** 고쳐서 얻을 게 실제로 있는가. false 면 처방을 띄우지 않는다. */
  actionable: boolean
  advice: string
}

export interface TideBreakdown {
  failed?: false
  /** 이 분석 자체의 매칭 기준. overall.map50 과 미세하게 다를 수 있다. */
  baseline_map50: number | null
  baseline_classes: number[]
  params: {
    collection_conf: number
    deploy_conf: number
    t_fg: number
    t_bg: number
    metric: string
  }
  errors: TideError[]
  per_class_counts: { cls: number; name: string; counts: Record<TideKind, number> }[]
  confusion_pairs: { pred: string; gt: string; count: number }[]
  note: string
}

/** 분해만 실패한 경우. 키가 아예 없는 것(=예전 리포트)과 구분해야 한다. */
export interface TideFailure {
  failed: true
  message: string
}

export interface LabelFinding {
  kind: string
  label: string
  cls: number
  name: string
  conf: number | null
  iou: number
  score: number
  /** 의심 지점. [x1, y1, x2, y2], 각각 0~1. */
  box: [number, number, number, number]
  /** 판정의 근거가 된 상대 박스. 없을 수 있다. */
  ref_box: [number, number, number, number] | null
  ref_name: string | null
  message: string
}

export interface LabelIssues {
  available: boolean
  /** 모델 근거를 쓰지 못한 사유. 쓸 수 있으면 null. */
  reason: string | null
  /** false 면 라벨만 보고 찾은 것(겹치는 정답 박스)만 실려 있다. */
  model_evidence: boolean
  /** 상한을 적용하기 전 전체 건수. */
  total: number
  shown: number
  images_cap: number
  kinds: { kind: string; label: string; count: number }[]
  /** 검증 셋만 봤다는 한계. 화면 맨 위에 그대로 띄운다. */
  scope_note: string
  items: {
    image: string
    name: string
    width: number
    height: number
    findings: LabelFinding[]
    gt: AnalysisBox[]
    pred: AnalysisBox[]
  }[]
}

/** 한 섹션만 실패한 경우. 키가 아예 없는 것(=예전 리포트)과 구분해야 한다. */
export interface QualitySectionFailure {
  failed: true
  message: string
}

export interface DuplicateGroup {
  size: number
  /**
   * exact = 파일이 완전히 같다 / near = 같은 사진의 다른 사본 (둘 다 지워도 된다)
   * similar = 닮았지만 같다고 단정 못 함 / chain = 일부 쌍만 확정 (유사도는 전이적이지 않다)
   */
  kind: 'exact' | 'near' | 'similar' | 'chain'
  images: { path: string; split: 'train' | 'val' }[]
}

export interface LeakPair {
  train: string
  val: string
  hamming: number
  /** 모델 특징을 못 쓴 경우 null. */
  cosine: number | null
  ncc: number
  exact: boolean
}

export interface QualityReport {
  schema_version: number
  dataset_id: string
  created_at: number
  elapsed_s: number
  params: {
    imgsz: number
    device: string
    hamming: number
    cosine: number
    ncc: number
    delete_ncc: number
    /** true 가 아니면 모델 특징 없이 밝기 패턴만으로 판정했다는 뜻이다. */
    embedding: true | { used: false; reason: string }
  }
  counts: {
    train: number
    val: number
    scanned: number
    unreadable: number
    candidate_pairs: number
  }
  duplicates:
    | (QualitySectionFailure | never)
    | {
        failed?: false
        /** 지워도 학습에 쓰이는 사진이 그대로인 장수. 완전한 묶음에서만 센다. */
        wasted: number
        image_total: number
        group_total: number
        groups_cap: number
        groups: DuplicateGroup[]
        message: string
      }
  leakage:
    | (QualitySectionFailure | never)
    | {
        failed?: false
        val_leaked: number
        val_total: number
        ratio: number
        exact_pairs: number
        pair_total: number
        pairs_cap: number
        pairs: LeakPair[]
        message: string
      }
  imbalance:
    | (QualitySectionFailure | never)
    | {
        failed?: false
        ratio: number | null
        classes: {
          cls: number
          name: string
          train_instances: number
          val_instances: number
          train_images: number
          val_images: number
        }[]
        missing_in_train: string[]
        missing_in_val: string[]
        rare_in_train: string[]
      }
  /** 이 검사가 무엇을 보지 않았는가. 화면에 그대로 띄운다. */
  notes: string[]
}

/** 서버가 요청 때마다 만들어 얹는다. 리포트 파일에는 없다. */
export interface NextAction {
  code: string
  severity: 'critical' | 'warn' | 'info'
  title: string
  cause: string
  fix: string
}

export interface AnalysisReport {
  schema_version: number
  run_id: string
  weights: string
  device: string
  elapsed_s: number
  classes: string[]
  overall: {
    images: number
    instances: number
    precision: number | null
    recall: number | null
    map50: number | null
    map50_95: number | null
  }
  per_class: {
    cls: number
    name: string
    images: number
    instances: number
    precision: number | null
    recall: number | null
    f1: number | null
    ap50: number | null
    ap50_95: number | null
    /** false 면 검증 셋에 이 클래스의 정답이 없어 성능을 알 수 없다는 뜻이다. */
    evaluated: boolean
  }[]
  /** schema_version 2 부터. 그 이전 리포트에는 아예 없다. */
  tide?: TideBreakdown | TideFailure
  /** schema_version 3 부터. */
  label_issues?: LabelIssues
  /** 파일에는 없다 — API 가 요청마다 계산해 얹는다. */
  next_actions?: NextAction[]
  worst_classes: { name: string | null; ap50_95: number | null; message: string }[]
  conf_recommendation: {
    conf: number | null
    f1: number | null
    precision: number | null
    recall: number | null
    /** false 면 모델이 덜 학습돼 임계값 0 이 최적으로 나온 것이다. 쓰면 안 된다. */
    reliable: boolean
    message: string | null
    per_class: { cls: number; name: string; conf: number | null; f1: number | null }[]
    curve: { conf: number | null; f1: number | null }[]
  }
  gallery: {
    image: string
    name: string
    score: number
    tp: number
    fp: number
    fn: number
    gt: AnalysisBox[]
    pred: AnalysisBox[]
  }[]
  gallery_total: number
  gallery_cap: number
  /** 갤러리를 어떤 신뢰도 기준으로 그렸는지. */
  gallery_conf: number
}

export interface Estimate {
  ok: boolean
  reason?: string
  epoch_time_s: number
  total_time_s: number
  /** [낙관, 비관]. 점 추정만 보여주면 안 된다 — 보정 표본이 없을 때 특히. */
  range_s: [number, number]
  batch_effective: number
  vram_gb: number | null
  vram_total_gb: number | null
  vram_level: 'ok' | 'tight' | 'over'
  /** calibrated = 이 PC 의 실측으로 보정함. analytic = 근사식뿐. */
  source: 'calibrated' | 'analytic'
  samples: number
  assumptions: string[]
  warnings: { code: string; message: string; patch: Record<string, unknown> }[]
}

export interface Diagnosis {
  run_id: string
  status: RunStatus
  /** 규칙에 걸렸는가. false 면 원문과 로그만 보여준다. */
  matched: boolean
  code: string | null
  title: string | null
  cause: string | null
  fix: string | null
  /** 규칙에 걸린 로그 줄. */
  evidence: string[]
  log_tail: string[]
  /**
   * 같은 설정으로 다시 돌려서 결과가 달라질 여지가 없으면 null 이다
   * (예: 라벨이 없는 경우 — 데이터를 고치기 전에는 똑같이 실패한다).
   */
  retry: {
    label: string
    changed: Record<string, DiagnosisChange>
    params: Record<string, unknown>
    options: Record<string, unknown>
    devices: number[]
  } | null
}

export interface TrainEvent {
  t: 'start' | 'batch' | 'epoch' | 'final_val' | 'artifact' | 'checkpoint' | 'end' | 'warning'
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
  /** 이 run 이 잡아 본 VRAM 최대치(GB). CPU 학습이면 null. */
  mem_gb?: number | null
  /** NaN/Inf 라서 값을 싣지 못한 지표 키. 값 대신 사실만 남긴다. */
  nonfinite?: string[]
  loss_nan?: boolean
  /** t === 'warning' 일 때의 사람이 읽는 문장. 판정도 문장도 백엔드가 만든다. */
  message?: string
  /** 경고 종류. run 당 code 하나만 나온다. */
  code?: string
  severity?: 'info' | 'warn' | 'critical'
  /** 그래서 무엇을 하라는 한 줄. */
  hint?: string
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
