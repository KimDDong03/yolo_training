import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Dataset, DatasetReview } from '../types'
import { BoxOverlay, type OverlayBox } from './BoxOverlay'
import { DatasetPathWarning } from './DatasetPathWarning'
import { Modal } from './ui/Dialog'
import { EmptyState } from './ui/EmptyState'

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

  if (!dataset) return <EmptyState title="데이터셋 정보가 없습니다" />

  const report = dataset.report
  const problems = (review?.categories ?? []).filter((c) => c.total > 0)
  const stats = review?.box_stats

  return (
    <>
      <DatasetPathWarning dataset={dataset} />

      <div className="card">
        <h3>검수 요약</h3>
        <table>
          <caption className="sr-only">데이터셋 검수 요약</caption>
          <tbody>
            <tr><th scope="row">이미지</th><td>{report.total_images.toLocaleString()}장</td></tr>
            <tr>
              <th scope="row">train / val</th>
              <td>
                {report.train_count} / {report.val_count}{' '}
                {report.auto_split && <span className="muted">(자동 분할)</span>}
              </td>
            </tr>
            <tr><th scope="row">클래스</th><td>{dataset.classes.join(', ')}</td></tr>
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
          <div className="card-head">
            <h3>박스 분포</h3>
            <span className="muted small spacer" style={{ fontWeight: 400 }}>
              {stats.count.toLocaleString()}개 · 이미지 대비 1% 미만 {(stats.tiny_ratio * 100).toFixed(0)}%
            </span>
          </div>
          <Histogram title="크기 (이미지 면적 대비)" bins={stats.area} />
          <Histogram title="종횡비" bins={stats.aspect} />
          {stats.tiny_ratio > 0.3 && (
            <div className="help warn">
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
          <div className="row wrap tight" role="group" aria-label="문제 종류 고르기">
            <button className="chip" aria-pressed={!category} onClick={() => setCategory('')}>
              샘플 보기
            </button>
            {problems.map((c) => (
              <button
                key={c.code}
                className="chip"
                aria-pressed={category === c.code}
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
                {/* img onClick 은 키보드로 닿지 않는다. 버튼으로 감싸야 Tab·Enter 로 확대할 수 있다. */}
                <button
                  className="img-button"
                  aria-label={`${item.path.split('/').pop()} 확대`}
                  onClick={() => setZoom(api.datasetImageUrl(dataset.id, item.path))}
                >
                  <img className="preview-img" src={api.datasetImageUrl(dataset.id, item.path)} alt={item.path} />
                </button>
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

      {/* 손으로 만든 fixed 오버레이였다. <dialog> 는 ESC·포커스 트랩·포커스 복귀를 공짜로 준다. */}
      <Modal
        open={zoom !== null}
        onClose={() => setZoom(null)}
        className="dialog lightbox"
        label="확대한 이미지"
      >
        {zoom && (
          <button
            className="img-button"
            style={{ cursor: 'zoom-out' }}
            aria-label="확대 닫기"
            data-autofocus
            onClick={() => setZoom(null)}
          >
            <img src={zoom} alt="확대한 이미지" />
            <span className="lightbox-cap">{decodeURIComponent(zoom.split('/').pop() ?? '')} · 클릭하면 닫힙니다</span>
          </button>
        )}
      </Modal>
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
  // 라벨은 cx/cy/w/h 로 오고 오버레이는 xyxy 를 받는다.
  const overlay: OverlayBox[] = boxes.map((b) => ({
    box: [b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2],
    label: b.name,
    kind: 'gt',
  }))

  return (
    <BoxOverlay
      src={url}
      alt={path.split(/[\\/]/).pop() ?? path}
      boxes={overlay}
      onZoom={onZoom}
    />
  )
}
