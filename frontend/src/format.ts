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
