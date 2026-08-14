import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Dataset, DatasetReview } from '../types'

type SampleItem = { path: string; boxes: { name: string; cx: number; cy: number; w: number; h: number }[] }

export function DatasetReviewPanel({ dataset }: { dataset: Dataset | null | undefined }) {
  const [samples, setSamples] = useState<SampleItem[]>([])
  const [review, setReview] = useState<DatasetReview | null>(null)
  const [category, setCategory] = useState('')
  const [zoom, setZoom] = useState<string | null>(null)

  useEffect(() => {
    if (!dataset) return
    setCategory('')
    api.datasetSamples(dataset.id).then((r) => setSamples(r.samples)).catch(() => setSamples([]))
  }, [dataset?.id])

  useEffect(() => {
    if (!dataset) return
    api.datasetReview(dataset.id, category).then(setReview).catch(() => setReview(null))
  }, [dataset?.id, category])

  if (!dataset) return <p className="muted">데이터셋 정보가 없습니다.</p>

  const report = dataset.report
  const problems = (review?.categories ?? []).filter((c) => c.total > 0)
  const stats = review?.box_stats

  return (
    <>
      <div className="card">
        <h3>검수 요약</h3>
        <table>
          <tbody>
            <tr><th>이미지</th><td>{report.total_images.toLocaleString()}장</td></tr>
            <tr>
              <th>train / val</th>
              <td>
                {report.train_count} / {report.val_count}{' '}
                {report.auto_split && <span className="muted">(자동 분할)</span>}
              </td>
            </tr>
            <tr><th>클래스</th><td>{dataset.classes.join(', ')}</td></tr>
          </tbody>
        </table>

        {report.class_instances && (
          <div style={{ marginTop: 10 }}>
            {Object.entries(report.class_instances).map(([name, count]) => {
              const max = Math.max(...Object.values(report.class_instances!))
              return (
                <div key={name} className="row small" style={{ gap: 8, marginBottom: 4 }}>
                  <span style={{ width: 90 }} className="muted">{name}</span>
                  <div className="progress"><div style={{ width: `${(count / max) * 100}%` }} /></div>
                  <span style={{ width: 60, textAlign: 'right' }}>{count.toLocaleString()}</span>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {stats && stats.count > 0 && (
        <div className="card">
          <h3>
            박스 분포
            <span className="muted" style={{ float: 'right', fontWeight: 400 }}>
              {stats.count.toLocaleString()}개 · 이미지 대비 1% 미만 {(stats.tiny_ratio * 100).toFixed(0)}%
            </span>
          </h3>
          <Histogram title="크기 (이미지 면적 대비)" bins={stats.area} />
          <Histogram title="종횡비" bins={stats.aspect} />
          {stats.tiny_ratio > 0.3 && (
            <div className="help" style={{ color: 'var(--warn)' }}>
              작은 객체가 많습니다. 이미지 크기(imgsz)를 키우지 않으면 잘 안 잡힐 수 있습니다.
            </div>
          )}
        </div>
      )}

      <div className="card">
        <h3>문제 있는 이미지</h3>
        {problems.length === 0 ? (
          <p className="muted small">발견된 문제가 없습니다.</p>
        ) : (
          <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
            <button
              style={{ fontSize: 12, padding: '3px 10px', borderColor: !category ? 'var(--accent)' : undefined }}
              onClick={() => setCategory('')}
            >
              샘플 보기
            </button>
            {problems.map((c) => (
              <button
                key={c.code}
                style={{ fontSize: 12, padding: '3px 10px', borderColor: category === c.code ? 'var(--accent)' : undefined }}
                onClick={() => setCategory(c.code)}
              >
                {c.label} {c.total.toLocaleString()}
              </button>
            ))}
          </div>
        )}
        {category && review && (
          <div className="small muted" style={{ marginTop: 8 }}>
            {review.page.truncated
              ? `${review.page.total.toLocaleString()}건 중 ${review.page.stored.toLocaleString()}건만 기록했습니다 (상한 ${review.review_cap.toLocaleString()})`
              : `${review.page.total.toLocaleString()}건`}
          </div>
        )}
      </div>

      <div className="gallery">
        {category
          ? (review?.page.items ?? []).map((item) => (
              <figure key={item.path}>
                <img
                  className="preview-img"
                  src={api.datasetImageUrl(dataset.id, item.path)}
                  alt={item.path}
                  onClick={() => setZoom(api.datasetImageUrl(dataset.id, item.path))}
                  style={{ cursor: 'zoom-in' }}
                />
                <figcaption>
                  {item.path.split('/').pop()}
                  {item.detail && <span className="error"> · {item.detail}</span>}
                </figcaption>
              </figure>
            ))
          : samples.map((s) => (
              <figure key={s.path}>
                <SampleImage datasetId={dataset.id} path={s.path} boxes={s.boxes} onZoom={setZoom} />
                <figcaption>{s.path.split(/[\\/]/).pop()} · {s.boxes.length}개 박스</figcaption>
              </figure>
            ))}
      </div>

      {zoom && (
        <div
          onClick={() => setZoom(null)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 50,
            display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out',
          }}
        >
          <img src={zoom} alt="확대" style={{ maxWidth: '90vw', maxHeight: '90vh', imageRendering: 'pixelated' }} />
        </div>
      )}
    </>
  )
}

function Histogram({ title, bins }: { title: string; bins: { label: string; count: number }[] }) {
  const max = Math.max(1, ...bins.map((b) => b.count))
  return (
    <div style={{ marginBottom: 10 }}>
      <div className="small muted" style={{ marginBottom: 4 }}>{title}</div>
      {bins.map((b) => (
        <div key={b.label} className="row small" style={{ gap: 8, marginBottom: 3 }}>
          <span style={{ width: 90 }} className="muted">{b.label}</span>
          <div className="progress"><div style={{ width: `${(b.count / max) * 100}%` }} /></div>
          <span style={{ width: 50, textAlign: 'right' }}>{b.count.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

function SampleImage({
  datasetId,
  path,
  boxes,
  onZoom,
}: {
  datasetId: string
  path: string
  boxes: { name: string; cx: number; cy: number; w: number; h: number }[]
  onZoom?: (url: string) => void
}) {
  const url = api.datasetImageUrl(datasetId, path)
  return (
    <div style={{ position: 'relative' }}>
      <img
        className="preview-img"
        src={url}
        alt={path}
        style={{ cursor: onZoom ? 'zoom-in' : undefined }}
        onClick={() => onZoom?.(url)}
      />
      {boxes.map((b, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: `${(b.cx - b.w / 2) * 100}%`,
            top: `${(b.cy - b.h / 2) * 100}%`,
            width: `${b.w * 100}%`,
            height: `${b.h * 100}%`,
            border: '1.5px solid #35c46b',
            borderRadius: 2,
          }}
          title={b.name}
        />
      ))}
    </div>
  )
}
