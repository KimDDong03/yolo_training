const STATUS_LABELS: Record<string, string> = {
  queued: '대기',
  running: '학습중',
  completed: '완료',
  stopped: '정지됨',
  failed: '실패',
}

export function statusLabel(status: string) {
  return STATUS_LABELS[status] ?? status
}

/** 남은 시간 · 경과 시간 공통. 초 단위로 받아 사람이 읽는 문자열로. */
export function formatDuration(seconds: number) {
  const s = Math.max(0, Math.round(seconds))
  const m = Math.floor(s / 60)
  const h = Math.floor(m / 60)
  if (h > 0) return `${h}시간 ${m % 60}분`
  if (m > 0) return `${m}분 ${s % 60}초`
  return `${s}초`
}

/** 사이드바처럼 폭이 좁은 자리의 경과 시간. `1h 23m` / `18m` / `45s`. */
export function formatShort(seconds: number) {
  const s = Math.max(0, Math.round(seconds))
  const m = Math.floor(s / 60)
  const h = Math.floor(m / 60)
  if (h > 0) return `${h}h ${m % 60}m`
  if (m > 0) return `${m}m`
  return `${s}s`
}

/**
 * 목록에서 "언제" 를 한 마디로. 오늘·어제는 날짜보다 이 말이 빨리 읽힌다.
 * 경계는 자정이다 — 23시간 전이어도 날짜가 넘었으면 어제다.
 */
export function relativeDay(seconds: number, now = Date.now()) {
  const then = new Date(seconds * 1000)
  const midnight = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const days = Math.round((midnight(new Date(now)) - midnight(then)) / 86400000)
  if (days <= 0) return '오늘'
  if (days === 1) return '어제'
  if (days < 7) return `${days}일 전`
  return then.toLocaleDateString('ko-KR')
}

/** 실패 사유는 첫 줄만 쓴다. 서버가 준 문장을 그대로 보여주고 원인을 지어내지 않는다. */
export function firstLine(text: string | null | undefined) {
  if (!text) return ''
  const line = text.split(/\r?\n/).map((l) => l.trim()).find((l) => l.length > 0)
  return line ?? ''
}
