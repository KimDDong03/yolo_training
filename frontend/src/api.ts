import type { Artifacts, Dataset, Gpu, ParamSchema, Run } from './types'

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

  runs: () => req<Run[]>('/api/runs'),
  run: (id: string) => req<Run>(`/api/runs/${id}`),
  createRun: (body: { dataset_id: string; name: string; devices: number[]; params: Record<string, unknown> }) =>
    req<Run>('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  stopRun: (id: string, mode: 'graceful' | 'force') =>
    req<Run>(`/api/runs/${id}/stop?mode=${mode}`, { method: 'POST' }),
  deleteRun: (id: string) => req<{ status: string }>(`/api/runs/${id}`, { method: 'DELETE' }),
  artifacts: (id: string) => req<Artifacts>(`/api/runs/${id}/artifacts`),
  fileUrl: (runId: string, path: string) => `/api/runs/${runId}/files/${path}`,
}
