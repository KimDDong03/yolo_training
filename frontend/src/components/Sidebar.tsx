import { useEffect, useMemo, useRef, useState } from 'react'
import { statusLabel } from '../format'
import { seriesColor } from '../theme'
import type { Run, View } from '../types'
import type { LoadStatus } from '../useResource'
import { EmptyState, SkeletonRows } from './ui/EmptyState'

/** CompareView 는 run 마다 상세 + 전체 이벤트를 병렬로 받는다. 상한이 없으면 그게 그대로 부하가 된다. */
export const COMPARE_LIMIT = 6

const STATUS_FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'running', label: '학습중' },
  { key: 'completed', label: '완료' },
  { key: 'failed', label: '실패' },
] as const

type StatusFilter = (typeof STATUS_FILTERS)[number]['key']
type Sort = 'recent' | 'name' | 'status'

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
 * 예전에는 헤더 <select> 하나가 run 전환을 다 맡아서, 무엇이 돌고 있는지 보려면
 * 목록을 펼쳐야 했고 "새 학습" 은 그 목록의 첫 항목으로 위장돼 있었다.
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
  const [status, setStatus] = useState<StatusFilter>('all')
  const [sort, setSort] = useState<Sort>('recent')
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

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const filtered = runs.filter(
      (r) =>
        (status === 'all' || r.status === status) &&
        (!needle || r.name.toLowerCase().includes(needle) || r.id.toLowerCase().includes(needle)),
    )
    const order = ['running', 'queued', 'failed', 'stopped', 'completed']
    return [...filtered].sort((a, b) => {
      if (sort === 'name') return a.name.localeCompare(b.name, 'ko')
      if (sort === 'status') return order.indexOf(a.status) - order.indexOf(b.status) || b.created_at - a.created_at
      return b.created_at - a.created_at
    })
  }, [runs, query, status, sort])

  const currentId = view.kind === 'run' ? view.id : null
  const atLimit = compare.length >= COMPARE_LIMIT

  return (
    <nav className="sidebar" aria-label="실행 목록">
      <div className="sidebar-head">
        <button
          className={view.kind === 'new' ? 'primary' : ''}
          onClick={() => onNavigate({ kind: 'new' })}
          aria-current={view.kind === 'new' ? 'page' : undefined}
        >
          ＋ 새 학습
        </button>

        <label className="sr-only" htmlFor="run-search">
          실행 검색
        </label>
        <input
          id="run-search"
          ref={searchRef}
          className="small"
          type="search"
          value={query}
          placeholder="이름으로 검색 (Ctrl+K)"
          onChange={(e) => setQuery(e.target.value)}
        />

        <div className="segmented" role="radiogroup" aria-label="상태로 거르기">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              role="radio"
              aria-checked={status === f.key}
              className="tiny"
              onClick={() => setStatus(f.key)}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="sidebar-scroll">
        <div className="row sidebar-section">
          <span>실행 {shown.length}개</span>
          <label className="sr-only" htmlFor="run-sort">
            정렬 기준
          </label>
          <select
            id="run-sort"
            className="tiny spacer"
            style={{ width: 78 }}
            value={sort}
            onChange={(e) => setSort(e.target.value as Sort)}
          >
            <option value="recent">최신순</option>
            <option value="name">이름순</option>
            <option value="status">상태순</option>
          </select>
        </div>

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

        {runsStatus === 'ready' && shown.length === 0 && (
          <EmptyState
            title={runs.length === 0 ? '아직 학습 기록이 없습니다' : '조건에 맞는 실행이 없습니다'}
            description={runs.length === 0 ? '＋ 새 학습으로 시작하세요.' : undefined}
          />
        )}

        <ul>
          {shown.map((r) => {
            const checked = compare.includes(r.id)
            const busy = r.status === 'running' || r.status === 'queued'
            return (
              <li key={r.id} className="run-item">
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={!checked && atLimit}
                  aria-label={`${r.name} 비교 대상`}
                  onChange={(e) => onToggleCompare(r.id, e.target.checked)}
                />
                <button
                  className="nav-item"
                  aria-current={currentId === r.id ? 'page' : undefined}
                  onClick={() => onNavigate({ kind: 'run', id: r.id })}
                >
                  {/*
                    비교에 넣은 run 은 상태색 대신 비교 팔레트 색으로 보여준다.
                    CompareView 도 같은 compare 배열의 인덱스로 색을 고르므로 차트 범례와 정확히 맞는다.
                  */}
                  <span
                    className={checked ? 'dot' : `dot ${r.status}`}
                    style={checked ? { background: seriesColor(compare.indexOf(r.id)) } : undefined}
                    aria-hidden="true"
                  />
                  <span style={{ minWidth: 0, flex: 1 }}>
                    <span className="run-name">{r.name}</span>
                    <span className="run-meta">
                      {statusLabel(r.status)} · {new Date(r.created_at * 1000).toLocaleDateString('ko-KR')}
                    </span>
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
          })}
        </ul>
      </div>

      <div className="sidebar-foot stack">
        {compare.length > 0 && (
          <div className="row tight">
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
          데이터셋 <span className="muted tiny spacer">{datasetCount}개</span>
        </button>
      </div>
    </nav>
  )
}
