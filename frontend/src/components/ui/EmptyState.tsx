import type { ReactNode } from 'react'

/**
 * "아직 없음" 과 "못 불러옴" 을 같은 화면으로 보여주면 서버 장애가 정상 상태로 위장된다.
 * tone 으로 둘을 갈라 놓고, 실패에는 반드시 다시 시도할 방법을 준다.
 */
export function EmptyState({
  title,
  description,
  action,
  tone = 'empty',
}: {
  title: string
  description?: ReactNode
  action?: ReactNode
  tone?: 'empty' | 'error'
}) {
  return (
    <div className="empty">
      <h4 className={tone === 'error' ? 'error' : undefined}>{title}</h4>
      {description && <p>{description}</p>}
      {action}
    </div>
  )
}

/** 첫 로딩 자리를 잡아 준다. 목록이 튀어 올라오는 것보다 덜 산만하다. */
export function SkeletonRows({ rows = 3 }: { rows?: number }) {
  return (
    <div className="stack" aria-hidden="true">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton" style={{ width: `${100 - i * 12}%` }} />
      ))}
    </div>
  )
}
