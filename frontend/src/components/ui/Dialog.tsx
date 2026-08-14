import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

/**
 * 네이티브 <dialog> 껍데기.
 *
 * ESC · 포커스 트랩 · 백드롭 · 닫을 때 포커스 되돌리기를 브라우저가 이미 한다.
 * 직접 구현하면 그 중 하나는 반드시 빠지므로 showModal() 에 맡긴다.
 * 열릴 때 [data-autofocus] 요소로 포커스를 옮긴다 — React 의 autoFocus 는
 * showModal() 보다 먼저 실행돼 자리를 잡지 못한다.
 */
export function Modal({
  open,
  onClose,
  className,
  labelledBy,
  label,
  children,
}: {
  open: boolean
  onClose: () => void
  className?: string
  /** 다이얼로그 안의 제목 요소 id. 제목이 없으면 label 을 쓴다. */
  labelledBy?: string
  label?: string
  children: ReactNode
}) {
  const ref = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (open) {
      if (!el.open) el.showModal()
      el.querySelector<HTMLElement>('[data-autofocus]')?.focus()
    } else if (el.open) {
      el.close()
    }
  }, [open])

  return (
    <dialog
      ref={ref}
      className={className}
      aria-labelledby={labelledBy}
      aria-label={label}
      onCancel={(e) => {
        e.preventDefault() // 닫기는 상태로만 한다. 안 그러면 open 과 실제 DOM 이 어긋난다.
        onClose()
      }}
      onClick={(e) => {
        if (e.target === ref.current) onClose() // 백드롭 클릭은 dialog 요소 자신에게 잡힌다
      }}
    >
      {children}
    </dialog>
  )
}

interface ConfirmOptions {
  title: string
  body?: string
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
}

interface PromptOptions {
  title: string
  label: string
  body?: string
  placeholder?: string
  initial?: string
  confirmLabel?: string
}

type Request = ({ kind: 'confirm' } & ConfirmOptions) | ({ kind: 'prompt' } & PromptOptions)

interface DialogApi {
  confirm: (options: ConfirmOptions) => Promise<boolean>
  prompt: (options: PromptOptions) => Promise<string | null>
}

const DialogContext = createContext<DialogApi>({
  confirm: async () => false,
  prompt: async () => null,
})

/** 되돌릴 수 없는 동작 앞에 쓴다. `if (!(await confirm({...}))) return` */
export function useConfirm() {
  return useContext(DialogContext).confirm
}

/** window.prompt 대체. 취소하면 null 이다. */
export function usePrompt() {
  return useContext(DialogContext).prompt
}

export function DialogProvider({ children }: { children: ReactNode }) {
  const [request, setRequest] = useState<Request | null>(null)
  const [text, setText] = useState('')
  const resolver = useRef<((value: boolean | string | null) => void) | null>(null)
  const titleId = useId()
  const inputId = useId()

  const settle = useCallback((value: boolean | string | null) => {
    const resolve = resolver.current
    resolver.current = null
    setRequest(null)
    resolve?.(value)
  }, [])

  const api = useMemo<DialogApi>(
    () => ({
      confirm: (options) =>
        new Promise<boolean>((resolve) => {
          resolver.current = resolve as (value: boolean | string | null) => void
          setRequest({ kind: 'confirm', ...options })
        }),
      prompt: (options) =>
        new Promise<string | null>((resolve) => {
          resolver.current = resolve as (value: boolean | string | null) => void
          setText(options.initial ?? '')
          setRequest({ kind: 'prompt', ...options })
        }),
    }),
    [],
  )

  const cancel = () => settle(request?.kind === 'prompt' ? null : false)

  return (
    <DialogContext.Provider value={api}>
      {children}
      <Modal open={request !== null} onClose={cancel} className="dialog" labelledBy={titleId}>
        {request && (
          <form
            onSubmit={(e) => {
              e.preventDefault()
              settle(request.kind === 'prompt' ? text.trim() : true)
            }}
          >
            <div className="dialog-body">
              <h2 id={titleId}>{request.title}</h2>
              {request.body && <p>{request.body}</p>}
              {request.kind === 'prompt' && (
                <div className="field">
                  <label htmlFor={inputId}>{request.label}</label>
                  <input
                    id={inputId}
                    data-autofocus
                    value={text}
                    placeholder={request.placeholder}
                    onChange={(e) => setText(e.target.value)}
                  />
                </div>
              )}
            </div>
            <div className="dialog-foot">
              <button type="button" onClick={cancel} data-autofocus={request.kind === 'confirm' || undefined}>
                {request.kind === 'confirm' ? (request.cancelLabel ?? '취소') : '취소'}
              </button>
              <button
                type="submit"
                className={request.kind === 'confirm' && request.danger ? 'danger' : 'primary'}
                disabled={request.kind === 'prompt' && !text.trim()}
              >
                {request.confirmLabel ?? '확인'}
              </button>
            </div>
          </form>
        )}
      </Modal>
    </DialogContext.Provider>
  )
}
