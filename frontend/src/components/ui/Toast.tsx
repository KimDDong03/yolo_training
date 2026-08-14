import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react'

type ToastKind = 'info' | 'success' | 'error'

interface Toast {
  id: number
  kind: ToastKind
  message: string
}

type Push = (message: string, kind?: ToastKind) => void

const ToastContext = createContext<Push>(() => {})

/**
 * 잠깐 뜨고 사라지는 알림.
 *
 * 사라져도 되는 것만 여기로 보낸다 — 복사 완료, 삭제 실패, 프리셋 저장 같은 일회성 결과.
 * 작업을 막는 오류(모델 경로가 틀렸다, 내보내기가 실패했다)는 화면에 남아야 하므로
 * 토스트로 옮기지 않는다. 지속되는 서버 연결 실패도 배너 쪽이다.
 */
export function useToast(): Push {
  return useContext(ToastContext)
}

const LIFETIME_MS: Record<ToastKind, number> = { info: 4000, success: 3500, error: 8000 }

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextId = useRef(1)

  const push = useCallback<Push>((message, kind = 'info') => {
    const id = nextId.current++
    setToasts((list) => [...list, { id, kind, message }])
    setTimeout(() => setToasts((list) => list.filter((t) => t.id !== id)), LIFETIME_MS[kind])
  }, [])

  const dismiss = (id: number) => setToasts((list) => list.filter((t) => t.id !== id))

  const polite = toasts.filter((t) => t.kind !== 'error')
  const assertive = toasts.filter((t) => t.kind === 'error')

  return (
    <ToastContext.Provider value={push}>
      {children}
      {/*
        두 라이브 영역을 항상 렌더한다. 영역이 알림보다 먼저 DOM 에 있어야
        스크린리더가 새로 들어온 내용을 읽는다 — 조건부로 만들면 첫 알림을 놓친다.
      */}
      <div className="toast-stack">
        <div className="stack" role="status" aria-live="polite">
          {polite.map((t) => (
            <ToastItem key={t.id} toast={t} onDismiss={dismiss} />
          ))}
        </div>
        <div className="stack" role="alert" aria-live="assertive">
          {assertive.map((t) => (
            <ToastItem key={t.id} toast={t} onDismiss={dismiss} />
          ))}
        </div>
      </div>
    </ToastContext.Provider>
  )
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: number) => void }) {
  return (
    <div className={`toast ${toast.kind}`}>
      <p>{toast.message}</p>
      <button className="ghost btn-xs" aria-label="알림 닫기" onClick={() => onDismiss(toast.id)}>
        ✕
      </button>
    </div>
  )
}
