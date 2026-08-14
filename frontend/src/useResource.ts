import { useCallback, useEffect, useRef, useState } from 'react'

export type LoadStatus = 'loading' | 'ready' | 'error'

export interface Resource<T> {
  data: T
  status: LoadStatus
  error: string
  /** 연속 실패 횟수. 0 이면 마지막 시도가 성공했다. */
  failures: number
  reload: () => Promise<void>
}

/**
 * 목록 요청 하나의 상태.
 *
 * 원래는 전부 `.catch(() => {})` 라서 서버가 죽은 것과 기록이 0개인 것이 똑같이 보였다.
 * 그래서 상태를 나눠 든다.
 *
 * 한 번 성공한 뒤의 실패는 status 를 'error' 로 떨어뜨리지 않는다 — 2초 폴링이
 * 한 번 삐끗할 때마다 화면을 비우면 못 쓴다. 대신 failures 를 올려서 호출부가
 * "몇 번 연속 실패했을 때" 배너를 띄울지 스스로 정하게 한다.
 */
export function useResource<T>(load: () => Promise<T>, initial: T, pollMs?: number): Resource<T> {
  const [data, setData] = useState<T>(initial)
  const [status, setStatus] = useState<LoadStatus>('loading')
  const [error, setError] = useState('')
  const [failures, setFailures] = useState(0)

  // load 는 호출부에서 매 렌더 새로 만들어지기 쉽다. ref 에 담아 폴링 타이머를 다시 걸지 않는다.
  const loadRef = useRef(load)
  loadRef.current = load

  // 응답이 요청 순서대로 오지 않는다. 마지막으로 시작한 요청만 반영한다 —
  // 이 데이터 위에 App 의 삭제 불변식이 얹혀 있어서, 늦게 온 옛 목록이 지워진 run 을
  // 되살리면 화면이 지워진 run 으로 되돌아간다.
  const seq = useRef(0)

  const reload = useCallback(async () => {
    const mine = ++seq.current
    try {
      const next = await loadRef.current()
      if (mine !== seq.current) return
      setData(next)
      setError('')
      setFailures(0)
      setStatus('ready')
    } catch (e) {
      if (mine !== seq.current) return
      setError(String(e instanceof Error ? e.message : e))
      setFailures((n) => n + 1)
      setStatus((s) => (s === 'ready' ? 'ready' : 'error'))
    }
  }, [])

  // setInterval 이 아니라 "끝난 뒤 다시 예약" 이다. 서버가 느려질 때 요청이 겹쳐 쌓이지 않는다.
  useEffect(() => {
    let stopped = false
    let timer: number | undefined
    const tick = async () => {
      await reload()
      if (stopped || !pollMs) return
      timer = window.setTimeout(tick, pollMs)
    }
    tick()
    return () => {
      stopped = true
      if (timer !== undefined) clearTimeout(timer)
    }
  }, [reload, pollMs])

  return { data, status, error, failures, reload }
}
