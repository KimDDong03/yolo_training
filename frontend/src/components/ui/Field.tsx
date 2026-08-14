import { useId, type ReactNode } from 'react'

export interface FieldStatus {
  kind: 'ok' | 'bad'
  text: ReactNode
}

interface ControlProps {
  id: string
  'aria-describedby': string | undefined
  'aria-invalid': true | undefined
}

/**
 * 라벨 · 도움말 · 검증 메시지를 컨트롤에 이어 붙이는 표현 전용 wrapper.
 *
 * 렌더 프롭으로 id/aria 를 넘기는 이유 — 원래 코드는 <label> 과 <input> 이 형제라
 * 연결이 끊겨 있었다. 연결을 호출부의 성의에 맡기면 또 끊긴다. 여기서 강제한다.
 *
 * 스키마 해석(ParamField 의 타입별 컨트롤 선택, 경로 후보 목록)은 여기 넣지 않는다.
 * 도메인 지식이 ui/ 로 새면 이 wrapper 를 다른 화면에서 못 쓴다.
 */
export function Field({
  label,
  labelExtra,
  help,
  status,
  children,
}: {
  label: ReactNode
  labelExtra?: ReactNode
  help?: ReactNode
  status?: FieldStatus
  children: (props: ControlProps) => ReactNode
}) {
  const id = useId()
  const helpId = `${id}-help`
  const statusId = `${id}-status`
  const describedBy = [status ? statusId : null, help ? helpId : null].filter(Boolean).join(' ')

  return (
    <div className="field">
      <label htmlFor={id}>
        {label}
        {labelExtra}
      </label>
      {children({
        id,
        'aria-describedby': describedBy || undefined,
        'aria-invalid': status?.kind === 'bad' ? true : undefined,
      })}
      {status && (
        <div id={statusId} className={`help ${status.kind}`} aria-live="polite">
          {status.text}
        </div>
      )}
      {help && (
        <div id={helpId} className="help">
          {help}
        </div>
      )}
    </div>
  )
}
