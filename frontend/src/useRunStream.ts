import { useEffect, useRef, useState } from 'react'
import type { TrainEvent } from './types'

export interface StreamState {
  events: TrainEvent[]
  batch: TrainEvent | null
  logs: string[]
  finished: boolean
  connected: boolean
}

const EMPTY: StreamState = { events: [], batch: null, logs: [], finished: false, connected: false }
const MAX_LOGS = 3000

/**
 * run 하나의 실시간 스트림.
 *
 * 접속하면 서버가 과거 이벤트 전체를 스냅샷으로 먼저 보내준다. 그래서 학습 도중에
 * 새로고침하거나 나중에 접속해도 차트가 처음부터 완전히 그려진다.
 * 스냅샷과 라이브 메시지가 겹칠 수 있으므로 (종류, 에폭, 시각)으로 중복을 흡수한다.
 */
export function useRunStream(runId: string | null): StreamState {
  const [state, setState] = useState<StreamState>(EMPTY)
  const seen = useRef<Set<string>>(new Set())

  useEffect(() => {
    seen.current = new Set()
    setState(EMPTY)
    if (!runId) return

    const key = (e: TrainEvent) => `${e.t}:${e.epoch ?? ''}:${e.ts}`
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const socket = new WebSocket(`${proto}://${location.host}/api/runs/${runId}/ws`)
    let closed = false

    socket.onopen = () => setState((s) => ({ ...s, connected: true }))
    socket.onclose = () => {
      if (!closed) setState((s) => ({ ...s, connected: false }))
    }
    socket.onmessage = (msg) => {
      const data = JSON.parse(msg.data)
      if (data.type === 'snapshot') {
        const events: TrainEvent[] = data.events ?? []
        events.forEach((e) => seen.current.add(key(e)))
        setState({
          events,
          batch: data.batch ?? null,
          logs: (data.logs ?? []).slice(-MAX_LOGS),
          finished: !!data.finished,
          connected: true,
        })
      } else if (data.type === 'event') {
        const event: TrainEvent = data.event
        if (event.t === 'batch') {
          setState((s) => ({ ...s, batch: event }))
          return
        }
        const k = key(event)
        if (seen.current.has(k)) return
        seen.current.add(k)
        setState((s) => ({
          ...s,
          events: [...s.events, event],
          finished: event.t === 'end' ? true : s.finished,
        }))
      } else if (data.type === 'log') {
        setState((s) => ({ ...s, logs: [...s.logs, ...data.lines].slice(-MAX_LOGS) }))
      }
    }

    return () => {
      closed = true
      socket.close()
    }
  }, [runId])

  return state
}
