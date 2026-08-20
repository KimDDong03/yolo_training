import { useEffect, useMemo, useRef, useState } from 'react'
import { firstLine, formatShort, relativeDay, statusLabel } from '../format'
import { seriesColor } from '../theme'
import type { Run, View } from '../types'
import type { LoadStatus } from '../useResource'
import { EmptyState, SkeletonRows } from './ui/EmptyState'

/** CompareView 는 run 마다 상세 + 전체 이벤트를 병렬로 받는다. 상한이 없으면 그게 그대로 부하가 된다. */
export const COMPARE_LIMIT = 6

const LIVE: string[] = ['running', 'queued']

interface Props {
  runs: Run[]
  runsStatus: LoadStatus
  onRetryRuns: () => void
  datasetCount: number
  view: View
  compare: string[]
  onNavigate: (view: View) => void
  onToggleCompare: (id: string, on: boolean) => void
  onClearCompare: () => void
  onDeleteRun: (run: Run) => void
}

/**
 * 항상 보이는 실행 목록.
 *
 * 목록을 "진행 중" 과 "지난 실행" 으로 가른다. 둘은 보는 이유가 다르다 — 앞은 지금 어떻게
 * 되고 있나이고, 뒤는 무엇이 나왔나다. 그래서 진행 중에는 진행바를, 지난 실행에는 결과를 붙인다.
 *
 * 상태 필터와 정렬 컨트롤은 뺐다. 두 구획 + 검색이 그 일을 대신하고, 컨트롤 세 개가
 * 목록보다 위에 쌓여 있으면 정작 무엇이 도는지가 안 보인다.
 */
export function Sidebar({
  runs,
  runsStatus,
  onRetryRuns,
  datasetCount,
  view,
  compare,
  onNavigate,
  onToggleCompare,
  onClearCompare,
  onDeleteRun,
}: Props) {
  const [query, setQuery] = useState('')
  const searchRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        searchRef.current?.focus()
        searchRef.current?.select()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const [live, past] = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const shown = runs.filter(
      (r) => !needle || r.name.toLowerCase().includes(needle) || r.id.toLowerCase().includes(needle),
    )
    const byRecent = (a: Run, b: Run) => b.created_at - a.created_at
    return [
      shown.filter((r) => LIVE.includes(r.status)).sort(byRecent),
      shown.filter((r) => !LIVE.includes(r.status)).sort(byRecent),
    ]
  }, [runs, query])

  const currentId = view.kind === 'run' ? view.id : null
  const atLimit = compare.length >= COMPARE_LIMIT
  // 체크박스는 실행 비교 화면에서만 나온다. 늘 켜 두면 목록의 모든 줄이 "고르는 줄" 로 보인다.
  const picking = view.kind === 'compare'
  const empty = live.length === 0 && past.length === 0

  function row(r: Run) {
    const checked = compare.includes(r.id)
    const busy = LIVE.includes(r.status)
    return (
      <li
        key={r.id}
        className={`run-item${busy ? ' live' : ''}${currentId === r.id ? ' current' : ''}`}
      >
        {picking && (
          <input
            type="checkbox"
            checked={checked}
            disabled={!checked && atLimit}
            aria-label={`${r.name} 비교 대상`}
            onChange={(e) => onToggleCompare(r.id, e.target.checked)}
          />
        )}
        <button
          className="nav-item"
          aria-current={currentId === r.id ? 'page' : undefined}
          onClick={() => onNavigate({ kind: 'run', id: r.id })}
        >
          {/*
            비교에 넣은 run 은 상태색 대신 비교 팔레트 색으로 보여준다.
            CompareView 도 같은 compare 배열의 인덱스로 색을 고르므로 차트 범례와 정확히 맞는다.
          */}
          {!busy && (
            <span
              className={checked ? 'dot' : `dot ${r.status}`}
              style={checked ? { background: seriesColor(compare.indexOf(r.id)) } : undefined}
              aria-hidden="true"
            />
          )}
          <span style={{ minWidth: 0, flex: 1 }}>
            <span className="run-name">{r.name}</span>
            {busy ? <LiveMeta run={r} /> : <PastMeta run={r} />}
          </span>
        </button>
        <button
          className="ghost btn-xs muted"
          aria-label={`${r.name} 삭제`}
          title={busy ? '학습 중에는 삭제할 수 없습니다' : '삭제'}
          disabled={busy}
          onClick={() => onDeleteRun(r)}
        >
          ✕
        </button>
      </li>
    )
  }

  return (
    <nav className="sidebar" aria-label="실행 목록">
      <div className="sidebar-head">
        <button
          className={view.kind === 'new' ? 'sidebar-new primary' : 'sidebar-new'}
          onClick={() => onNavigate({ kind: 'new' })}
          aria-current={view.kind === 'new' ? 'page' : undefined}
        >
          ＋ 새 학습
        </button>

        <label className="sr-only" htmlFor="run-search">
          실행 검색
        </label>
        <div className="sidebar-search">
          <svg className="icon" viewBox="0 0 13 13" fill="none" aria-hidden="true">
            <circle cx="5.5" cy="5.5" r="4" stroke="currentColor" strokeWidth="1.3" />
            <path d="M8.6 8.6L12 12" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
          </svg>
          <input
            id="run-search"
            ref={searchRef}
            className="small"
            type="search"
            value={query}
            placeholder="이름으로 검색"
            onChange={(e) => setQuery(e.target.value)}
          />
          <span className="keycap" aria-hidden="true">Ctrl K</span>
        </div>
      </div>

      <div className="sidebar-scroll">
        {runsStatus === 'loading' && <SkeletonRows rows={4} />}

        {runsStatus === 'error' && (
          <EmptyState
            tone="error"
            title="실행 목록을 불러오지 못했습니다"
            description="서버가 응답하지 않습니다."
            action={
              <button className="btn-sm" onClick={onRetryRuns}>
                다시 시도
              </button>
            }
          />
        )}

        {runsStatus === 'ready' && empty && (
          <EmptyState
            title={runs.length === 0 ? '아직 학습 기록이 없습니다' : '조건에 맞는 실행이 없습니다'}
            description={runs.length === 0 ? '＋ 새 학습으로 시작하세요.' : undefined}
          />
        )}

        {live.length > 0 && (
          <>
            <div className="sidebar-section">진행 중 · {live.length}</div>
            <ul>{live.map(row)}</ul>
          </>
        )}

        {past.length > 0 && (
          <>
            <div className="sidebar-section">지난 실행 · {past.length}</div>
            <ul>{past.map(row)}</ul>
          </>
        )}
      </div>

      <div className="sidebar-foot">
        {compare.length > 0 && (
          <div className="row tight" style={{ padding: '0 0 var(--sp-1)' }}>
            <button className="btn-sm" style={{ flex: 1 }} onClick={() => onNavigate({ kind: 'compare' })}>
              {compare.length}개 비교 보기
            </button>
            <button className="btn-xs ghost" aria-label="비교 선택 모두 해제" onClick={onClearCompare}>
              ✕
            </button>
          </div>
        )}
        <button
          className="nav-item"
          aria-current={view.kind === 'datasets' ? 'page' : undefined}
          onClick={() => onNavigate({ kind: 'datasets' })}
        >
          데이터셋 <span className="muted tiny spacer">{datasetCount}</span>
        </button>
        <button
          className="nav-item"
          aria-current={view.kind === 'compare' ? 'page' : undefined}
          onClick={() => onNavigate({ kind: 'compare' })}
        >
          실행 비교 <span className="muted tiny spacer">{compare.length}</span>
        </button>
      </div>
    </nav>
  )
}

/** 진행 중 — 진행바 + `37/100 · 18m`. 값은 서버 요약에서 온다. */
function LiveMeta({ run }: { run: Run }) {
  const { epoch, total_epochs: total } = run.summary ?? {}
  const elapsed = run.started_at ? Date.now() / 1000 - run.started_at : null
  const parts = [
    epoch != null && total ? `${epoch}/${total}` : statusLabel(run.status),
    elapsed != null ? formatShort(elapsed) : null,
  ].filter(Boolean)

  return (
    <>
      {epoch != null && total ? (
        <div
          className="progress thin"
          style={{ marginTop: 6 }}
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={total}
          aria-valuenow={epoch}
        >
          <div style={{ width: `${Math.min(100, (epoch / total) * 100)}%` }} />
        </div>
      ) : null}
      <span className="run-meta">{parts.join(' · ')}</span>
    </>
  )
}

/**
 * 지난 실행 — 무엇이 나왔는지 한 줄.
 *
 * 실패는 서버가 저장한 사유의 첫 줄만 쓴다. 워커가 비정상 종료하면 그 자리에 일반적인
 * 종료 코드 문구만 들어 있는데, 그걸 그럴듯한 원인으로 바꿔 쓰지 않는다. 상세는 실패 진단이 한다.
 */
function PastMeta({ run }: { run: Run }) {
  const when = relativeDay(run.finished_at ?? run.created_at)
  const best = run.summary?.best_map
  const epoch = run.summary?.epoch

  if (run.status === 'failed') {
    const reason = firstLine(run.error)
    return <span className="run-meta err">{reason || `실패 · ${when}`}</span>
  }
  if (run.status === 'stopped') {
    return <span className="run-meta">{epoch != null ? `중지됨 · ${epoch}에폭` : `중지됨 · ${when}`}</span>
  }
  if (best != null) {
    return <span className="run-meta">mAP50-95 {best.toFixed(3)} · {when}</span>
  }
  return <span className="run-meta">{statusLabel(run.status)} · {when}</span>
}
