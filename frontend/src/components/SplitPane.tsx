import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'

const MIN = 25
const MAX = 75
const DEFAULT = 55
const STEP = 2

const clamp = (v: number) => Math.min(MAX, Math.max(MIN, v))

/**
 * 좌우 분할. 비율은 브라우저에 남는다.
 *
 * 55/45 고정이던 자리다 — 로그를 넓게 볼지 예측 이미지를 크게 볼지는 그때그때 다르다.
 * 핸들은 WAI-ARIA window splitter 패턴을 따라 포커스를 받고 화살표로도 움직인다.
 * 마우스로만 되는 리사이저는 없느니만 못하다.
 */
export function SplitPane({
  storageKey,
  label,
  left,
  right,
}: {
  storageKey: string
  label: string
  left: ReactNode
  right: ReactNode
}) {
  const [ratio, setRatio] = useState(() => {
    const saved = Number(localStorage.getItem(storageKey))
    return Number.isFinite(saved) && saved >= MIN && saved <= MAX ? saved : DEFAULT
  })
  const [dragging, setDragging] = useState(false)
  const boxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    localStorage.setItem(storageKey, String(Math.round(ratio)))
  }, [storageKey, ratio])

  const applyFromX = useCallback((clientX: number) => {
    const box = boxRef.current?.getBoundingClientRect()
    if (!box || box.width === 0) return
    setRatio(clamp(((clientX - box.left) / box.width) * 100))
  }, [])

  function onKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    let next = ratio
    if (e.key === 'ArrowLeft') next = ratio - STEP
    else if (e.key === 'ArrowRight') next = ratio + STEP
    else if (e.key === 'Home') next = MIN
    else if (e.key === 'End') next = MAX
    else return
    e.preventDefault()
    setRatio(clamp(next))
  }

  return (
    <div className={`split ${dragging ? 'dragging' : ''}`} ref={boxRef}>
      <div className="split-pane" style={{ width: `${ratio}%` }}>
        {left}
      </div>
      <div
        className="splitter"
        role="separator"
        tabIndex={0}
        aria-orientation="vertical"
        aria-label={label}
        aria-valuemin={MIN}
        aria-valuemax={MAX}
        aria-valuenow={Math.round(ratio)}
        aria-valuetext={`왼쪽 ${Math.round(ratio)}%`}
        onKeyDown={onKeyDown}
        onDoubleClick={() => setRatio(DEFAULT)}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          setDragging(true)
        }}
        onPointerMove={(e) => {
          if (e.currentTarget.hasPointerCapture(e.pointerId)) applyFromX(e.clientX)
        }}
        onPointerUp={(e) => {
          if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId)
          setDragging(false)
        }}
        // 창 전환이나 시스템 제스처로 캡처를 뺏길 수 있다. pointerup 만 보면
        // dragging 이 켜진 채 남아 텍스트 선택이 막히고 커서가 col-resize 로 굳는다.
        onPointerCancel={() => setDragging(false)}
        onLostPointerCapture={() => setDragging(false)}
      />
      <div className="split-pane" style={{ flex: 1 }}>
        {right}
      </div>
    </div>
  )
}
