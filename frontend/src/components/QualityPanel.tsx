import { useEffect, useMemo, useState } from 'react'

import { api } from '../api'
import type { Dataset, DuplicateGroup, JobStatus, LeakPair, QualityReport } from '../types'
import { DatasetPathWarning } from './DatasetPathWarning'
import { DuplicateCompare, type CompareImage } from './DuplicateCompare'
import { useConfirm } from './ui/Dialog'
import { useToast } from './ui/Toast'

const POLL_MS = 2000

/** 지워도 되는 묶음과 눈으로 확인해야 하는 묶음. 사용자에게는 이 둘만 구분되면 된다. */
const DELETABLE = new Set<DuplicateGroup['kind']>(['exact', 'near'])

const KIND_LABEL: Record<DuplicateGroup['kind'], string> = {
  exact: '파일까지 완전히 같음',
  near: '같은 사진의 다른 사본',
  similar: '닮았지만 같다고 단정 못 함',
  chain: '일부만 이어져 있음',
}

type Opened = { kind: 'group'; index: number } | { kind: 'pair'; index: number }

function name(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

/**
 * 묶음에서 지울 후보. 한 장만 남기고 나머지다.
 *
 * 남기는 한 장은 **train 쪽을 먼저** 고른다 — val 사본을 지우면 누수까지 같이 사라진다.
 * 같은 split 이면 **경로가 짧은 쪽**을 남긴다. 사본은 이름을 덧붙여 만들어지므로
 * (`a.jpg` → `a_copy.jpg`) 거의 항상 원본이 짧다.
 *
 * localeCompare 를 쓰지 않는다 — 밑줄·마침표를 낮은 가중치로 보는 탓에 `a_copy.jpg` 가
 * `a.jpg` 보다 앞서고, 그러면 사본을 남기고 원본을 지운다.
 *
 * 지워도 된다고 판정된 묶음만 대상이다. similar/chain 은 눈으로 봐야 하므로 비워 둔다.
 */
/** 이 장만 남고 묶음의 나머지가 전부 지움으로 찍혀 있는가. */
function isKeeper(marked: Set<string>, all: string[], path: string): boolean {
  if (all.length < 2 || marked.has(path)) return false
  return all.every((p) => p === path || marked.has(p))
}

function proposeForGroup(group: DuplicateGroup): string[] {
  if (!DELETABLE.has(group.kind)) return []
  const ordered = [...group.images].sort((a, b) => {
    if (a.split !== b.split) return a.split === 'train' ? -1 : 1
    if (a.path.length !== b.path.length) return a.path.length - b.path.length
    return a.path < b.path ? -1 : a.path > b.path ? 1 : 0
  })
  return ordered.slice(1).map((image) => image.path)
}

/**
 * 데이터 품질 검사 — 중복 · train/val 누수 · 클래스 불균형.
 *
 * 누수가 이 화면의 이유다. 검증용 사진이 학습용에 섞이면 모델이 외운 사진으로 채점하게 되어
 * mAP 가 실제보다 높게 나온다. 진단 화면의 숫자가 전부 그 위에 서 있다.
 *
 * 판정만 보여주고 끝내면 사용자는 파일 탐색기를 열어 손으로 지운 뒤 다시 등록해야 한다.
 * 그래서 여기서 바로 고르고 비교하고 지운다 — **원본까지 지우므로 되돌릴 수 없다.**
 */
export function QualityPanel({
  dataset,
  onDatasetChanged,
}: {
  dataset: Dataset | null | undefined
  /** 장수·검수 경고가 바뀌므로 목록 쪽 상태도 다시 읽어야 한다. */
  onDatasetChanged?: () => void
}) {
  const confirm = useConfirm()
  const toast = useToast()
  const [job, setJob] = useState<JobStatus | null>(null)
  const [report, setReport] = useState<QualityReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [marked, setMarked] = useState<Set<string>>(new Set())
  const [failed, setFailed] = useState<{ path: string; error: string }[]>([])
  const [opened, setOpened] = useState<Opened | null>(null)

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
    setMarked(new Set())
    setFailed([])
    setOpened(null)
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

  const dup = report?.duplicates
  const leak = report?.leakage
  const bal = report?.imbalance
  const groups = dup && !('failed' in dup && dup.failed) ? dup.groups : []
  const pairs = leak && !('failed' in leak && leak.failed) ? leak.pairs : []

  // 같은 검증용 사진이 여러 쌍에 걸린다. 경로로 한 번만 센다.
  const leakedVal = useMemo(
    () => Array.from(new Set(pairs.map((p) => p.val))),
    [pairs],
  )

  if (!dataset) return null

  const start = async () => {
    setBusy(true)
    setError('')
    // 캐시가 있으면 재검사가 1초 안에 끝난다. 낡은 리포트를 띄워 둔 채로 "검사 완료" 를
    // 보여 주면 지금 결과로 오해한다.
    setReport(null)
    setMarked(new Set())
    setOpened(null)
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

  const toggle = (path: string) =>
    setMarked((prev) => {
      const next = new Set(prev)
      if (!next.delete(path)) next.add(path)
      return next
    })

  const add = (paths: string[]) =>
    setMarked((prev) => {
      const next = new Set(prev)
      paths.forEach((p) => next.add(p))
      return next
    })

  const clear = (paths: string[]) =>
    setMarked((prev) => {
      const next = new Set(prev)
      paths.forEach((p) => next.delete(p))
      return next
    })

  /**
   * 이 한 장만 남기고 묶음의 나머지를 전부 고른다.
   *
   * 이미 그 한 장만 남아 있는데 또 누르면 묶음 선택을 통째로 푼다. 같은 버튼으로 되돌릴 수
   * 없으면 잘못 고른 뒤 빠져나갈 길이 없다.
   */
  const keepOnly = (all: string[], keeper: string) =>
    setMarked((prev) => {
      const next = new Set(prev)
      const others = all.filter((p) => p !== keeper)
      if (isKeeper(prev, all, keeper)) {
        all.forEach((p) => next.delete(p))
      } else {
        next.delete(keeper)
        others.forEach((p) => next.add(p))
      }
      return next
    })

  const removeSelected = async () => {
    const paths = Array.from(marked)
    if (!paths.length) return
    // 한 묶음을 통째로 고르면 그 사진이 데이터셋에서 아예 사라진다. 막지는 않되 말은 한다.
    const wiped = groups.filter(
      (g) => g.images.length > 0 && g.images.every((i) => marked.has(i.path)),
    ).length
    const ok = await confirm({
      title: `${paths.length}장을 지울까요?`,
      body:
        `원본 폴더(${dataset.root})의 이미지 파일과 짝 라벨을 함께 지웁니다.` +
        ' 되돌릴 수 없습니다.' +
        (wiped
          ? ` 그중 ${wiped}개 묶음은 남는 장이 없어 그 사진이 통째로 사라집니다.`
          : ''),
      confirmLabel: '지우기',
      danger: true,
    })
    if (!ok) return

    setBusy(true)
    setError('')
    try {
      const result = await api.deleteDatasetImages(dataset.id, paths)
      setFailed(result.failed)
      setMarked(new Set())
      setOpened(null)
      toast(
        `${result.deleted}장을 지웠습니다. 학습 ${result.train_count.toLocaleString()} /` +
          ` 검증 ${result.val_count.toLocaleString()}`,
        'success',
      )
      onDatasetChanged?.()
      // 지운 사진이 실린 리포트는 서버가 이미 버렸다. 캐시가 남아 재검사는 금방 끝난다.
      await start()
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
      setBusy(false)
    }
  }

  const running = job?.status === 'running'
  const events = job?.events ?? []
  const stage = events.length ? events[events.length - 1].message : undefined

  const compare = ((): {
    title: string
    subtitle?: string
    images: CompareImage[]
    step: (delta: number) => void
  } | null => {
    if (!opened) return null
    if (opened.kind === 'group') {
      const group = groups[opened.index]
      if (!group) return null
      return {
        title: KIND_LABEL[group.kind],
        subtitle: `${group.size}장 · 묶음 ${opened.index + 1}/${groups.length}`,
        images: group.images.map((image) => ({ path: image.path, split: image.split })),
        step: (delta) =>
          setOpened({
            kind: 'group',
            index: (opened.index + delta + groups.length) % groups.length,
          }),
      }
    }
    const pair = pairs[opened.index]
    if (!pair) return null
    // 코사인은 모델 특징을 못 쓴 검사에서 null 이다. 없는 값을 지어내지 않는다.
    const note = pair.exact
      ? '파일 동일'
      : `NCC ${pair.ncc.toFixed(4)}` +
        (pair.cosine !== null ? ` · 코사인 ${pair.cosine.toFixed(4)}` : '')
    return {
      title: '검증용 사진이 학습용에도 있음',
      subtitle: `쌍 ${opened.index + 1}/${pairs.length}`,
      images: [
        { path: pair.train, split: 'train', note },
        { path: pair.val, split: 'val', note },
      ],
      step: (delta) =>
        setOpened({
          kind: 'pair',
          index: (opened.index + delta + pairs.length) % pairs.length,
        }),
    }
  })()

  const comparePaths = compare?.images.map((image) => image.path) ?? []

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
        {failed.length > 0 && (
          <div className="help warn" style={{ marginTop: 8 }}>
            지우지 못한 것이 {failed.length}건 있습니다.
            <ul className="small" style={{ margin: '4px 0 0', paddingLeft: 18 }}>
              {failed.map((f) => (
                <li key={f.path}>
                  {name(f.path)} — {f.error}
                </li>
              ))}
            </ul>
          </div>
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
                {pairs.length > 0 && (
                  <>
                    <div className="row small" style={{ gap: 8, marginTop: 8 }}>
                      <span className="muted">
                        {leak.pair_total > pairs.length
                          ? `${leak.pair_total.toLocaleString()}쌍 중 ${pairs.length}쌍`
                          : `${leak.pair_total.toLocaleString()}쌍`}
                        {leak.exact_pairs > 0 && ` · 그중 ${leak.exact_pairs}쌍은 파일까지 같습니다`}
                      </span>
                      {canShowImages && (
                        <button
                          className="btn-sm spacer"
                          onClick={() => add(leakedVal)}
                          disabled={busy}
                        >
                          겹치는 검증용 {leakedVal.length}장 고르기
                        </button>
                      )}
                    </div>
                    {pairs.map((pair, index) => (
                      <PairRow
                        key={`${pair.train}|${pair.val}`}
                        datasetId={dataset.id}
                        pair={pair}
                        canShowImages={canShowImages}
                        marked={marked}
                        onOpen={() => setOpened({ kind: 'pair', index })}
                        onKeepOnly={keepOnly}
                      />
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
                {dup.group_total > groups.length && (
                  <p className="small muted" style={{ marginTop: 8 }}>
                    묶음 {dup.group_total.toLocaleString()}개 중 {groups.length}개만 보여줍니다.
                  </p>
                )}
                {groups.some((g) => DELETABLE.has(g.kind)) && canShowImages && (
                  <button
                    className="btn-sm"
                    style={{ marginTop: 8 }}
                    disabled={busy}
                    onClick={() => add(groups.flatMap(proposeForGroup))}
                  >
                    지워도 되는 묶음에서 한 장씩만 남기기
                  </button>
                )}
                {/* 크기 2 짜리 묶음이 수십 개면 화면이 길어져 아래 카드가 밀린다.
                    판단에 필요한 값(요약 문장과 위 건수)은 밖에 두고 목록만 접는다.
                    6 은 잰 값이 아니라 화면 길이 판단이다 — 실측 근거는 없다. */}
                {groups.length > 0 && (
                  <details open={groups.length <= 6} style={{ marginTop: 10 }}>
                    <summary className="small muted">중복 묶음 {groups.length}개 보기</summary>
                    {groups.map((group, index) => (
                      <GroupRow
                        key={index}
                        datasetId={dataset.id}
                        group={group}
                        canShowImages={canShowImages}
                        marked={marked}
                        busy={busy}
                        onOpen={() => setOpened({ kind: 'group', index })}
                        onKeepOnly={(path) =>
                          keepOnly(
                            group.images.map((image) => image.path),
                            path,
                          )
                        }
                        onClear={() => clear(group.images.map((image) => image.path))}
                        onPropose={() => add(proposeForGroup(group))}
                      />
                    ))}
                  </details>
                )}
              </>
            ) : null}
          </div>

          {canShowImages && (
            <div className="card">
              <div className="row" style={{ gap: 8 }}>
                <div>
                  <strong>{marked.size}장</strong>
                  <span className="muted small">을 지우기로 골랐습니다</span>
                  <p className="small muted" style={{ margin: '2px 0 0' }}>
                    원본 폴더의 이미지와 라벨을 함께 지웁니다. 되돌릴 수 없습니다.
                  </p>
                </div>
                <button
                  className="btn-sm spacer"
                  onClick={() => setMarked(new Set())}
                  disabled={busy || marked.size === 0}
                >
                  선택 해제
                </button>
                <button
                  className="btn-sm danger"
                  onClick={removeSelected}
                  disabled={busy || marked.size === 0}
                >
                  {busy ? '처리 중…' : `${marked.size}장 지우기`}
                </button>
              </div>
            </div>
          )}

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

      {compare && (
        <DuplicateCompare
          datasetId={dataset.id}
          title={compare.title}
          subtitle={compare.subtitle}
          images={compare.images}
          marked={marked}
          keeperPath={comparePaths.find((p) => isKeeper(marked, comparePaths, p)) ?? null}
          onToggle={toggle}
          // 지금 화면에 안 보이는 장까지 포함해 묶음 전체에 적용한다.
          onKeepOnly={(path) => keepOnly(comparePaths, path)}
          onClose={() => setOpened(null)}
          onStep={compare.step}
        />
      )}
    </>
  )
}

function GroupRow({
  datasetId,
  group,
  canShowImages,
  marked,
  busy,
  onOpen,
  onKeepOnly,
  onClear,
  onPropose,
}: {
  datasetId: string
  group: DuplicateGroup
  canShowImages: boolean
  marked: Set<string>
  busy: boolean
  onOpen: () => void
  onKeepOnly: (path: string) => void
  onClear: () => void
  onPropose: () => void
}) {
  const paths = group.images.map((image) => image.path)
  const chosen = paths.filter((p) => marked.has(p)).length
  const wipesGroup = chosen === paths.length && chosen > 0

  return (
    <div style={{ marginTop: 10 }}>
      <div className="row small" style={{ gap: 8 }}>
        <span className="muted">
          {DELETABLE.has(group.kind) ? `${group.size - 1}장 지워도 됨` : '눈으로 확인'} ·{' '}
          {KIND_LABEL[group.kind]} · {group.size}장
        </span>
        {canShowImages && (
          <>
            <button className="btn-sm spacer" onClick={onOpen}>
              비교
            </button>
            {DELETABLE.has(group.kind) && (
              <button className="btn-sm" onClick={onPropose} disabled={busy}>
                한 장만 남기기
              </button>
            )}
            {chosen > 0 && (
              <button className="btn-sm" onClick={onClear} disabled={busy}>
                선택 해제
              </button>
            )}
          </>
        )}
      </div>
      {/* 유사도는 전이적이지 않다. A~B 와 B~C 만 확정돼도 셋이 한 묶음이 되는데(quality.py 의
          is_complete) A 와 C 는 다른 사진일 수 있다. 한 장만 남기라고 권할 수 없는 자리다. */}
      {group.kind === 'chain' && canShowImages && (
        <p className="small warn" style={{ margin: '4px 0 0' }}>
          모든 쌍이 같다고 확인되지는 않았습니다. 남길 한 장을 정하기 전에 각각 비교하세요.
        </p>
      )}
      {wipesGroup && (
        <p className="small error" style={{ margin: '4px 0 0' }}>
          남는 장이 없습니다 — 이 사진이 데이터셋에서 통째로 사라집니다.
        </p>
      )}
      <div className="row small wrap" style={{ gap: 8, marginTop: 4 }}>
        {group.images.map((image) =>
          canShowImages ? (
            <Thumb
              key={image.path}
              datasetId={datasetId}
              path={image.path}
              tag={image.split === 'train' ? '학습' : '검증'}
              marked={marked.has(image.path)}
              keeper={isKeeper(marked, paths, image.path)}
              onOpen={onOpen}
              onKeepOnly={() => onKeepOnly(image.path)}
            />
          ) : (
            <span key={image.path} className="muted">
              {name(image.path)}
            </span>
          ),
        )}
      </div>
    </div>
  )
}

function PairRow({
  datasetId,
  pair,
  canShowImages,
  marked,
  onOpen,
  onKeepOnly,
}: {
  datasetId: string
  pair: LeakPair
  canShowImages: boolean
  marked: Set<string>
  onOpen: () => void
  onKeepOnly: (all: string[], keeper: string) => void
}) {
  const paths = [pair.train, pair.val]
  // 쌍의 두 장을 모두 고른 경우. 이 둘은 언제나 같은 중복 묶음 안에 있으므로(둘 다 확정쌍에서
  // 나온다) 묶음에 다른 사본이 더 있으면 사진은 남는다 — "사라진다" 고 단정하지 않는다.
  // 묶음 줄이 상한(GROUPS_CAP)에 잘려 안 보일 수 있어서 여기서도 한 번 말한다.
  const bothChosen = paths.every((p) => marked.has(p))
  return (
    <div className="row small wrap" style={{ gap: 8, marginTop: 8 }}>
      {canShowImages ? (
        <>
          <Thumb
            datasetId={datasetId}
            path={pair.train}
            tag="학습"
            marked={marked.has(pair.train)}
            keeper={isKeeper(marked, paths, pair.train)}
            onOpen={onOpen}
            onKeepOnly={() => onKeepOnly(paths, pair.train)}
          />
          <Thumb
            datasetId={datasetId}
            path={pair.val}
            tag="검증"
            marked={marked.has(pair.val)}
            keeper={isKeeper(marked, paths, pair.val)}
            onOpen={onOpen}
            onKeepOnly={() => onKeepOnly(paths, pair.val)}
          />
        </>
      ) : (
        <span className="muted">
          학습: {name(pair.train)} · 검증: {name(pair.val)}
        </span>
      )}
      <span className="muted">{pair.exact ? '파일 동일' : `유사도 ${pair.ncc.toFixed(4)}`}</span>
      {canShowImages && (
        <button className="btn-sm spacer" onClick={onOpen}>
          비교
        </button>
      )}
      {bothChosen && (
        <p className="small warn" style={{ margin: '4px 0 0', width: '100%' }}>
          이 쌍의 두 장을 모두 골랐습니다. 같은 사진의 다른 사본이 없다면 이 사진은 사라집니다.
        </p>
      )}
    </div>
  )
}

/**
 * 묶음 안의 한 장.
 *
 * 버튼은 "이것만 남기기" 하나다 — 사용자가 실제로 하려는 일이 그것이고, 한 장씩 찍게 하면
 * 5장짜리 묶음에서 네 번을 눌러야 한다. 개별 남김/지움은 비교창에 둔다.
 * 상태는 색으로만 말하지 않는다 — 캡션에 글자로도 적는다.
 */
function Thumb({
  datasetId,
  path,
  tag,
  marked,
  keeper,
  onOpen,
  onKeepOnly,
}: {
  datasetId: string
  path: string
  tag: string
  marked: boolean
  keeper: boolean
  onOpen: () => void
  onKeepOnly: () => void
}) {
  const url = api.datasetImageUrl(datasetId, path)
  const state = marked ? 'thumb-marked' : keeper ? 'thumb-keeper' : undefined
  return (
    <figure style={{ margin: 0 }} className={state}>
      <button className="img-button" aria-label={`${name(path)} 비교`} onClick={onOpen}>
        <img className="preview-img" src={url} alt={name(path)} style={{ maxWidth: 120 }} />
      </button>
      <figcaption className="small muted">
        {tag} · {name(path)}
        {marked && <span className="error"> · 지움</span>}
      </figcaption>
      {/* 한 줄에 같은 글자의 버튼이 여러 개 선다. 이름에 파일명을 넣어야 어느 장의
          버튼인지 소리로도 갈린다. */}
      {/* 눌린 모양은 aria-pressed 규칙에 맡긴다. .primary 를 붙이면 그 규칙이 글자색을
          액센트로 되돌려 금색 배경 위에 금색 글자가 된다. */}
      <button
        className="btn-sm"
        style={{ marginTop: 2 }}
        onClick={onKeepOnly}
        aria-pressed={keeper}
        aria-label={`${name(path)} 이것만 남기기`}
      >
        {keeper ? '남길 장' : '이것만 남기기'}
      </button>
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
