import { useEffect, useMemo, useRef, useState } from 'react'

interface Props {
  lines: string[]
}

type Filter = 'all' | 'warn' | 'error'

export function LogView({ lines }: Props) {
  const [filter, setFilter] = useState<Filter>('all')
  const [follow, setFollow] = useState(true)
  const boxRef = useRef<HTMLDivElement>(null)

  const shown = useMemo(() => {
    if (filter === 'all') return lines
    const needles = filter === 'warn' ? ['warning', 'warn', '경고'] : ['error', 'traceback', 'exception', '오류']
    return lines.filter((l) => needles.some((n) => l.toLowerCase().includes(n)))
  }, [lines, filter])

  useEffect(() => {
    if (follow && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight
  }, [shown, follow])

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, marginBottom: 0 }}>
      <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        학습 로그
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          {(['all', 'warn', 'error'] as Filter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{ padding: '2px 8px', fontSize: 11, borderColor: filter === f ? 'var(--accent)' : undefined }}
            >
              {f === 'all' ? '전체' : f === 'warn' ? '경고' : '오류'}
            </button>
          ))}
          <label className="small muted row" style={{ gap: 4 }}>
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
            따라가기
          </label>
        </span>
      </h3>
      <div className="log" ref={boxRef}>
        {shown.length === 0 ? (
          <span className="muted">로그가 아직 없습니다.</span>
        ) : (
          shown.map((line, i) => {
            const low = line.toLowerCase()
            const cls = low.includes('error') || low.includes('traceback') ? 'err' : low.includes('warning') ? 'warn' : ''
            return (
              <div key={i} className={cls}>
                {line}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
