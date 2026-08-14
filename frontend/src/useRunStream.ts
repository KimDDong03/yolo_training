import { useCallback, useEffect, useRef, useState } from 'react'
import type { TrainEvent } from './types'

export type StreamStatus = 'connecting' | 'open' | 'closed' | 'gone'

interface StreamData {
  events: TrainEvent[]
  batch: TrainEvent | null
  logs: string[]
  finished: boolean
}

export interface StreamState extends StreamData {
  status: StreamStatus
  /** 연속 재연결 시도 횟수. 0 이면 마지막 연결이 성공했다. */
  attempt: number
  reconnect: () => void
}

const EMPTY: StreamData = { events: [], batch: null, logs: [], finished: false }

/**
 * 서버가 스냅샷으로 돌려주는 로그 줄 수와 맞춘다 (backend/app/services/event_stream.py 의 LOG_TAIL_LINES).
 * 프론트가 더 많이 들고 있으면 재연결할 때마다 스냅샷이 그만큼을 잘라내 로그가 거꾸로 짧아진다.
 */
const MAX_LOGS = 2000

const BACKOFF_MS = [1000, 2000, 4000, 8000, 15000]

/** 서버가 "그런 run 없다"며 닫는 코드. 다시 붙어도 답이 같으므로 재시도하지 않는다. */
const CLOSE_RUN_GONE = 4404

/**
 * run 하나의 실시간 스트림.
 *
 * 접속하면 서버가 과거 이벤트 전체를 스냅샷으로 먼저 보내준다. 그래서 학습 도중에
 * 새로고침하거나 나중에 접속해도 차트가 처음부터 완전히 그려진다.
 * 스냅샷과 라이브 메시지가 겹칠 수 있으므로 (종류, 에폭, 시각)으로 중복을 흡수한다.
 *
 * 끊기면 지수 백오프로 다시 붙는다 — 예전에는 끊긴 채로 "연결 끊김"만 띄우고
 * 새로고침 말고는 복구할 방법이 없었다. 백엔드를 재시작해도 스냅샷으로 복원된다.
 */
export function useRunStream(runId: string | null): StreamState {
  const [data, setData] = useState<StreamData>(EMPTY)
  const [status, setStatus] = useState<StreamStatus>('closed')
  const [attempt, setAttempt] = useState(0)
  const [nonce, setNonce] = useState(0)

  const seen = useRef<Set<string>>(new Set())
  const finishedRef = useRef(false)

  const reconnect = useCallback(() => setNonce((n) => n + 1), [])

  useEffect(() => {
    seen.current = new Set()
    finishedRef.current = false
    setData(EMPTY)
    setAttempt(0)

    if (!runId) {
      setStatus('closed')
      return
    }

    const key = (e: TrainEvent) => `${e.t}:${e.epoch ?? ''}:${e.ts}`
    let socket: WebSocket | null = null
    let timer: number | undefined
    let disposed = false
    let tries = 0

    const connect = () => {
      if (disposed) return
      setStatus('connecting')
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      socket = new WebSocket(`${proto}://${location.host}/api/runs/${runId}/ws`)

      socket.onopen = () => {
        if (disposed) return
        tries = 0
        setAttempt(0)
        setStatus('open')
      }

      socket.onmessage = (msg) => {
        if (disposed) return
        const message = JSON.parse(msg.data)
        if (message.type === 'snapshot') {
          // 스냅샷은 authoritative replace 다 — 서버가 파일을 처음부터 다시 읽어 만든다.
          const events: TrainEvent[] = message.events ?? []
          seen.current = new Set(events.map(key))
          finishedRef.current = !!message.finished
          setData({
            events,
            batch: message.batch ?? null,
            logs: (message.logs ?? []).slice(-MAX_LOGS),
            finished: !!message.finished,
          })
        } else if (message.type === 'event') {
          const event: TrainEvent = message.event
          if (event.t === 'batch') {
            setData((s) => ({ ...s, batch: event }))
            return
          }
          const k = key(event)
          if (seen.current.has(k)) return
          seen.current.add(k)
          if (event.t === 'end') finishedRef.current = true
          setData((s) => ({
            ...s,
            events: [...s.events, event],
            finished: event.t === 'end' ? true : s.finished,
          }))
        } else if (message.type === 'log') {
          setData((s) => ({ ...s, logs: [...s.logs, ...message.lines].slice(-MAX_LOGS) }))
        }
      }

      socket.onclose = (e) => {
        if (disposed) return
        if (e.code === CLOSE_RUN_GONE) {
          setStatus('gone')
          return
        }
        // 학습이 끝난 run 이면 더 올 것이 없다. 빈 소켓을 15초마다 다시 열 이유가 없다.
        if (finishedRef.current) {
          setStatus('closed')
          return
        }
        setStatus('closed')
        const delay = BACKOFF_MS[Math.min(tries, BACKOFF_MS.length - 1)]
        tries += 1
        setAttempt(tries)
        timer = window.setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      disposed = true
      if (timer !== undefined) clearTimeout(timer)
      socket?.close()
    }
  }, [runId, nonce])

  return { ...data, status, attempt, reconnect }
}
