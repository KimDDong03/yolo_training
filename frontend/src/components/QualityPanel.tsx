import { useEffect, useState } from 'react'

import { api } from '../api'
import type { Dataset, DuplicateGroup, JobStatus, QualityReport } from '../types'
import { DatasetPathWarning } from './DatasetPathWarning'
import { Modal } from './ui/Dialog'

const POLL_MS = 2000

/** 지워도 되는 묶음과 눈으로 확인해야 하는 묶음. 사용자에게는 이 둘만 구분되면 된다. */
const DELETABLE = new Set<DuplicateGroup['kind']>(['exact', 'near'])

const KIND_LABEL: Record<DuplicateGroup['kind'], string> = {
  exact: '파일까지 완전히 같음',
  near: '같은 사진의 다른 사본',
  similar: '닮았지만 같다고 단정 못 함',
  chain: '일부만 이어져 있음',
}

function name(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

/**
 * 데이터 품질 검사 — 중복 · train/val 누수 · 클래스 불균형.
 *
 * 누수가 이 화면의 이유다. 검증용 사진이 학습용에 섞이면 모델이 외운 사진으로 채점하게 되어
 * mAP 가 실제보다 높게 나온다. 진단 화면의 숫자가 전부 그 위에 서 있다.
 */
export function QualityPanel({ dataset }: { dataset: Dataset | null | undefined }) {
  const [job, setJob] = useState<JobStatus | null>(null)
  const [report, setReport] = useState<QualityReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [zoom, setZoom] = useState<string | null>(null)

  const id = dataset?.id
  // 경로가 낡았으면 이미지 API 가 반드시 403 이다. 깨진 사진 수십 장을 그리는 대신
  // 배너와 파일명만 보여준다. root 경계를 느슨하게 하지는 않는다.
  const canShowImages = !dataset?.path_status || dataset.path_status.ok

  useEffect(() => {
    if (!id) return
    let cancelled = false
    setJob(null)
    setReport(null)
    setError('')
    api.qualityStatus(id).then((s) => !cancelled && setJob(s)).catch(() => {})
    api.qualityReport(id).then((r) => !cancelled && setReport(r)).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [id])

  useEffect(() => {
    if (!id || job?.status !== 'running') return
    let cancelled = false
    const timer = setInterval(async () => {
      try {
        const next = await api.qualityStatus(id)
        if (cancelled) return
        setJob(next)
        if (next.status === 'completed') setReport(await api.qualityReport(id))
      } catch {
        /* 폴링 실패는 다음 주기에 다시 시도한다 */
      }
    }, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [job?.status, id])

  if (!dataset) return null

  const start = async () => {
    setBusy(true)
    setError('')
    // 캐시가 있으면 재검사가 1초 안에 끝난다. 낡은 리포트를 띄워 둔 채로 "검사 완료" 를
    // 보여 주면 지금 결과로 오해한다.
    setReport(null)
    try {
      const next = await api.startQuality(dataset.id, { imgsz: 224, use_gpu: false })
      setJob(next)
      // completed 일 때만 받아 온다. 잡 시작은 quality.json 을 지우지 않으므로,
      // 곧바로 실패한 잡에서 리포트를 받으면 지난번 결과를 이번 결과로 보여 주게 된다.
      if (next.status === 'completed') setReport(await api.qualityReport(dataset.id))
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  const running = job?.status === 'running'
  const events = job?.events ?? []
  const stage = events.length ? events[events.length - 1].message : undefined

  const dup = report?.duplicates
  const leak = report?.leakage
  const bal = report?.imbalance

  return (
    <>
      <div className="card">
        <div className="card-head">
          <h3>데이터 품질</h3>
          <button className="btn-sm spacer" onClick={start} disabled={busy || running}>
            {running ? '검사 중…' : report ? '다시 검사' : '검사 시작'}
          </button>
        </div>
        <p className="help">
          같은 사진이 중복으로 들어갔는지, 검증용 사진이 학습용에 섞였는지, 클래스가 한쪽으로
          쏠렸는지 봅니다. CPU 로 돌아 학습의 GPU 를 뺏지 않습니다.
        </p>
        {running && <p className="small muted">{stage ?? '준비 중…'}</p>}
        {error && <p className="error small">{error}</p>}
        {job?.status === 'failed' && (
          <p className="error small">
            검사가 실패했습니다: {String(job.result?.error ?? job.error ?? '')}
          </p>
        )}
        {report && (
          <p className="small muted">
            {report.counts.scanned.toLocaleString()}장 검사 (학습 {report.counts.train.toLocaleString()} /
            검증 {report.counts.val.toLocaleString()}) · {report.elapsed_s}초
            {report.counts.unreadable > 0 && ` · 열지 못한 파일 ${report.counts.unreadable}장`}
          </p>
        )}
      </div>

      {!report ? null : (
        <>
          <DatasetPathWarning dataset={dataset} />

          {/* 누수를 맨 위에 둔다 — 진단 화면의 mAP 가 전부 이것 위에 서 있다. */}
          <div className="card">
            <h3>검증용 사진의 오염</h3>
            {leak && 'failed' in leak && leak.failed ? (
              <p className="error small">{leak.message}</p>
            ) : leak && !('failed' in leak && leak.failed) ? (
              <>
                <div className={leak.ratio >= 0.01 ? 'help warn' : 'help'}>{leak.message}</div>
                {leak.pairs.length > 0 && (
                  <>
                    <p className="small muted" style={{ marginTop: 8 }}>
                      {leak.pair_total > leak.pairs.length
                        ? `${leak.pair_total.toLocaleString()}쌍 중 ${leak.pairs.length}쌍`
                        : `${leak.pair_total.toLocaleString()}쌍`}
                      {leak.exact_pairs > 0 && ` · 그중 ${leak.exact_pairs}쌍은 파일까지 같습니다`}
                    </p>
                    {leak.pairs.map((p) => (
                      <div key={`${p.train}|${p.val}`} className="row small" style={{ gap: 8, marginTop: 8 }}>
                        {canShowImages ? (
                          <>
                            <Thumb datasetId={dataset.id} path={p.train} tag="학습" onZoom={setZoom} />
                            <Thumb datasetId={dataset.id} path={p.val} tag="검증" onZoom={setZoom} />
                          </>
                        ) : (
                          <span className="muted">
                            학습: {name(p.train)} · 검증: {name(p.val)}
                          </span>
                        )}
                        <span className="muted">{p.exact ? '파일 동일' : `유사도 ${p.ncc.toFixed(4)}`}</span>
                      </div>
                    ))}
                  </>
                )}
              </>
            ) : null}
          </div>

          <div className="card">
            <h3>중복된 사진</h3>
            {dup && 'failed' in dup && dup.failed ? (
              <p className="error small">{dup.message}</p>
            ) : dup && !('failed' in dup && dup.failed) ? (
              <>
                <div className={dup.wasted > 0 ? 'help warn' : 'help'}>{dup.message}</div>
                {dup.group_total > dup.groups.length && (
                  <p className="small muted" style={{ marginTop: 8 }}>
                    묶음 {dup.group_total.toLocaleString()}개 중 {dup.groups.length}개만 보여줍니다.
                  </p>
                )}
                {/* 크기 2 짜리 묶음이 수십 개면 화면이 길어져 아래 카드가 밀린다.
                    판단에 필요한 값(요약 문장과 위 건수)은 밖에 두고 목록만 접는다.
                    6 은 잰 값이 아니라 화면 길이 판단이다 — 실측 근거는 없다. */}
                {dup.groups.length > 0 && (
                  <details open={dup.groups.length <= 6} style={{ marginTop: 10 }}>
                    <summary className="small muted">
                      중복 묶음 {dup.groups.length}개 보기
                    </summary>
                    {dup.groups.map((g, gi) => (
                      <div key={gi} style={{ marginTop: 10 }}>
                        <div className="small muted">
                          {DELETABLE.has(g.kind) ? `${g.size - 1}장 지워도 됨` : '눈으로 확인'} ·{' '}
                          {KIND_LABEL[g.kind]} · {g.size}장
                        </div>
                        <div className="row small wrap" style={{ gap: 8, marginTop: 4 }}>
                          {g.images.map((im) =>
                            canShowImages ? (
                              <Thumb
                                key={im.path}
                                datasetId={dataset.id}
                                path={im.path}
                                tag={im.split === 'train' ? '학습' : '검증'}
                                onZoom={setZoom}
                              />
                            ) : (
                              <span key={im.path} className="muted">
                                {name(im.path)}
                              </span>
                            ),
                          )}
                        </div>
                      </div>
                    ))}
                  </details>
                )}
              </>
            ) : null}
          </div>

          <div className="card">
            <h3>클래스 균형</h3>
            {bal && 'failed' in bal && bal.failed ? (
              <p className="error small">{bal.message}</p>
            ) : bal && !('failed' in bal && bal.failed) ? (
              <>
                {bal.ratio !== null && bal.ratio >= 10 && (
                  <div className="help warn">
                    가장 많은 클래스가 가장 적은 클래스보다 {bal.ratio}배 많습니다. 적은 쪽은 잘
                    학습되지 않을 수 있습니다.
                  </div>
                )}
                {bal.missing_in_val.length > 0 && (
                  <div className="help warn">
                    검증용에 정답이 하나도 없는 클래스가 있습니다({bal.missing_in_val.join(', ')}).
                    이 클래스의 성능은 측정할 수 없습니다.
                  </div>
                )}
                {bal.rare_in_train.length > 0 && (
                  <div className="help warn">
                    학습용 정답이 20개 미만인 클래스가 있습니다({bal.rare_in_train.join(', ')}).
                  </div>
                )}
                <ClassBars rows={bal.classes} />
              </>
            ) : null}
          </div>

          <div className="card">
            <h3>이 검사가 보지 않은 것</h3>
            <ul className="small muted" style={{ margin: 0, paddingLeft: 18 }}>
              {report.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
              {report.params.embedding !== true && (
                <li className="error">{report.params.embedding.reason}</li>
              )}
            </ul>
          </div>
        </>
      )}

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
          </button>
        )}
      </Modal>
    </>
  )
}

function Thumb({
  datasetId,
  path,
  tag,
  onZoom,
}: {
  datasetId: string
  path: string
  tag: string
  onZoom: (url: string) => void
}) {
  const url = api.datasetImageUrl(datasetId, path)
  return (
    <figure style={{ margin: 0 }}>
      <button className="img-button" aria-label={`${name(path)} 확대`} onClick={() => onZoom(url)}>
        <img className="preview-img" src={url} alt={name(path)} style={{ maxWidth: 120 }} />
      </button>
      <figcaption className="small muted">
        {tag} · {name(path)}
      </figcaption>
    </figure>
  )
}

function ClassBars({
  rows,
}: {
  rows: { name: string; train_instances: number; val_instances: number }[]
}) {
  const max = Math.max(1, ...rows.map((r) => r.train_instances))
  return (
    <div style={{ marginTop: 10 }}>
      {rows.map((r) => (
        <div key={r.name} className="row small" style={{ gap: 8, marginBottom: 4 }}>
          <span style={{ width: 90 }} className="muted">
            {r.name}
          </span>
          <div className="progress">
            <div style={{ width: `${(r.train_instances / max) * 100}%` }} />
          </div>
          <span style={{ width: 110, textAlign: 'right' }}>
            학습 {r.train_instances.toLocaleString()} / 검증 {r.val_instances.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  )
}
