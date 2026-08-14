import type {
  AnalysisReport,
  Artifacts,
  Dataset,
  DatasetReview,
  Diagnosis,
  Estimate,
  ExportStatus,
  Gpu,
  JobStatus,
  ModelCheck,
  ParamSchema,
  PredictResult,
  Preset,
  Recommendation,
  Run,
  SystemInfo,
  TrainEvent,
  WeightCandidate,
} from './types'

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* 응답이 JSON 이 아니면 상태 코드만 쓴다 */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  paramsSchema: () => req<{ schema: ParamSchema; presets: Record<string, Record<string, unknown>> }>('/api/params/schema'),
  gpus: () => req<{ gpus: Gpu[] }>('/api/system/gpus'),
  systemInfo: () => req<SystemInfo>('/api/system/info'),
  weightCandidates: () => req<{ candidates: WeightCandidate[] }>('/api/system/weights'),
  validateModel: (model: string) =>
    req<ModelCheck>('/api/system/validate-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    }),

  presets: () => req<{ presets: Preset[] }>('/api/presets'),
  savePreset: (name: string, params: Record<string, unknown>, options: Record<string, unknown>) =>
    req<Preset>('/api/presets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, params, options }),
    }),
  deletePreset: (name: string) =>
    req<{ status: string }>(`/api/presets/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  datasets: () => req<Dataset[]>('/api/datasets'),
  registerPath: (path: string, name: string, valRatio: number) =>
    req<Dataset>('/api/datasets/path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, name, val_ratio: valRatio }),
    }),
  uploadZip: (file: File, name: string, valRatio: number) => {
    const form = new FormData()
    form.append('file', file)
    form.append('name', name)
    form.append('val_ratio', String(valRatio))
    return req<Dataset>('/api/datasets/upload', { method: 'POST', body: form })
  },
  deleteDataset: (id: string) => req<{ status: string }>(`/api/datasets/${id}`, { method: 'DELETE' }),
  datasetSamples: (id: string) =>
    req<{ samples: { path: string; boxes: { cls: number; name: string; cx: number; cy: number; w: number; h: number }[] }[] }>(
      `/api/datasets/${id}/samples`,
    ),
  datasetImageUrl: (id: string, path: string) => `/api/datasets/${id}/image?path=${encodeURIComponent(path)}`,
  datasetReview: (id: string, category = '', offset = 0, limit = 24) =>
    req<DatasetReview>(
      `/api/datasets/${id}/review?category=${encodeURIComponent(category)}&offset=${offset}&limit=${limit}`,
    ),

  runs: () => req<Run[]>('/api/runs'),
  run: (id: string) => req<Run>(`/api/runs/${id}`),
  events: (id: string) => req<{ events: TrainEvent[] }>(`/api/runs/${id}/events`),
  createRun: (body: {
    dataset_id: string
    name: string
    devices: number[]
    params: Record<string, unknown>
    options: Record<string, unknown>
    /** 이 실행이 어떤 실패한 실행의 재시도인지. */
    retry_of?: string
  }) =>
    req<Run>('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  diagnosis: (id: string) => req<Diagnosis>(`/api/runs/${id}/diagnosis`),
  startAnalysis: (id: string, body: { imgsz: number; batch: number; use_gpu: boolean }) =>
    req<JobStatus>(`/api/jobs/run/${id}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  analysisStatus: (id: string) => req<JobStatus>(`/api/jobs/run/${id}/analyze`),
  analysisReport: (id: string) => req<AnalysisReport>(`/api/runs/${id}/analysis/report`),
  recommendation: (datasetId: string, params: Record<string, unknown>, devices: number[]) =>
    req<Recommendation>(`/api/datasets/${datasetId}/recommendation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params, devices }),
    }),
  estimate: (datasetId: string, params: Record<string, unknown>, devices: number[]) =>
    req<Estimate>('/api/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: datasetId, params, devices }),
    }),
  stopRun: (id: string, mode: 'graceful' | 'force') =>
    req<Run>(`/api/runs/${id}/stop?mode=${mode}`, { method: 'POST' }),
  deleteRun: (id: string) => req<{ status: string }>(`/api/runs/${id}`, { method: 'DELETE' }),
  artifacts: (id: string) => req<Artifacts>(`/api/runs/${id}/artifacts`),
  startExport: (id: string, body: { format: string; weights: string; imgsz: number; half: boolean }) =>
    req<ExportStatus>(`/api/runs/${id}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  exportStatus: (id: string) => req<ExportStatus>(`/api/runs/${id}/export`),
  runWeights: (id: string) =>
    req<{ weights: { value: string; label: string; size_mb: number }[] }>(`/api/runs/${id}/weights`),
  predict: (
    id: string,
    file: File,
    opts: { weights: string; conf: number; iou: number; imgsz: number },
  ) => {
    const form = new FormData()
    form.append('file', file)
    form.append('weights', opts.weights)
    form.append('conf', String(opts.conf))
    form.append('iou', String(opts.iou))
    form.append('imgsz', String(opts.imgsz))
    return req<PredictResult>(`/api/runs/${id}/predict`, { method: 'POST', body: form })
  },
  fileUrl: (runId: string, path: string) => `/api/runs/${runId}/files/${path}`,
}
