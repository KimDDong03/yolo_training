import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react'
import { useToast } from './ui/Toast'

interface Props {
  lines: string[]
}

type Filter = 'all' | 'warn' | 'error'

const FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'warn', label: '경고' },
  { key: 'error', label: '오류' },
] as const

const NEEDLES: Record<Exclude<Filter, 'all'>, string[]> = {
  warn: ['warning', 'warn', '경고'],
  error: ['error', 'traceback', 'exception', '오류'],
}

export function LogView({ lines }: Props) {
  /*
   * 좁은 화면에서는 접힌 채로 시작한다. 로그가 118px 를 차지하면 차트 두 개가 밀려
   * "지금 잘 되고 있나" 에 답하는 화면이 로그 화면이 된다.
   */
  const [open, setOpen] = useState(() => !window.matchMedia('(max-width: 1439px)').matches)
  const [filter, setFilter] = useState<Filter>('all')
  const [query, setQuery] = useState('')
  const [follow, setFollow] = useState(true)
  const boxRef = useRef<HTMLDivElement>(null)
  const searchId = useId()
  const toast = useToast()

  const needle = query.trim().toLowerCase()

  const shown = useMemo(() => {
    let out = lines
    if (filter !== 'all') {
      out = out.filter((l) => NEEDLES[filter].some((n) => l.toLowerCase().includes(n)))
    }
    if (needle) out = out.filter((l) => l.toLowerCase().includes(needle))
    return out
  }, [lines, filter, needle])

  useEffect(() => {
    if (open && follow && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight
  }, [shown, follow, open])

  // 헤더에 띄우는 경고 수. 접혀 있어도 "볼 것이 있다" 는 보여야 한다.
  const warnCount = useMemo(
    () => lines.filter((l) => NEEDLES.warn.some((n) => l.toLowerCase().includes(n))).length,
    [lines],
  )

  return (
    <div className="card stack log-card" style={{ flex: open ? 1 : 'none', minHeight: 0, marginBottom: 0 }}>
      <div className="card-head">
        <button className="ghost log-toggle" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          <span aria-hidden="true">{open ? '▾' : '▸'}</span> 학습 로그
        </button>
        {warnCount > 0 && <span className="badge stopped">경고 {warnCount}</span>}
        <span className="muted tiny nowrap spacer">
          {shown.length === lines.length ? `${lines.length}줄` : `${shown.length}/${lines.length}줄`}
        </span>

        {!open ? null : (
        <>
        <label className="sr-only" htmlFor={searchId}>
          로그 내용 검색
        </label>
        <input
          id={searchId}
          className="tiny spacer"
          type="search"
          style={{ width: 130 }}
          placeholder="내용 검색"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <span className="segmented" role="radiogroup" aria-label="로그 종류로 거르기">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              role="radio"
              aria-checked={filter === f.key}
              className="tiny"
              onClick={() => setFilter(f.key)}
            >
              {f.label}
            </button>
          ))}
        </span>

        <label className="small muted row tight nowrap">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          따라가기
        </label>

        <button
          className="btn-xs"
          disabled={shown.length === 0}
          onClick={() => {
            navigator.clipboard
              ?.writeText(shown.join('\n'))
              .then(() => toast(`${shown.length}줄 복사됨`, 'success'))
              .catch(() => toast('클립보드에 복사하지 못했습니다', 'error'))
          }}
        >
          복사
        </button>
        </>
        )}
      </div>

      {/*
        role="log" 의 암묵 aria-live 는 polite 다 — 속성을 빼도 알림이 계속 나간다.
        따라가기를 끄면 사용자가 지나간 로그를 읽는 중이므로 명시적으로 off 해야 조용해진다.
      */}
      {open && (
      <div
        className="log"
        ref={boxRef}
        role="log"
        aria-label="학습 로그"
        aria-live={follow ? 'polite' : 'off'}
      >
        {shown.length === 0 ? (
          <span className="muted">{lines.length === 0 ? '로그가 아직 없습니다.' : '조건에 맞는 줄이 없습니다.'}</span>
        ) : (
          shown.map((line, i) => {
            const low = line.toLowerCase()
            const cls = low.includes('error') || low.includes('traceback') ? 'err' : low.includes('warning') ? 'warn' : ''
            return (
              <div key={i} className={cls}>
                {highlight(line, needle)}
              </div>
            )
          })
        )}
      </div>
      )}
    </div>
  )
}

/** 검색어에 걸린 구간을 <mark> 로 감싼다. 2000줄에서 눈으로 찾는 건 무리다. */
function highlight(line: string, needle: string): ReactNode {
  if (!needle) return line
  const low = line.toLowerCase()
  const parts: ReactNode[] = []
  let cursor = 0
  for (;;) {
    const at = low.indexOf(needle, cursor)
    if (at < 0) {
      parts.push(line.slice(cursor))
      break
    }
    if (at > cursor) parts.push(line.slice(cursor, at))
    parts.push(<mark key={at}>{line.slice(at, at + needle.length)}</mark>)
    cursor = at + needle.length
  }
  return parts
}
