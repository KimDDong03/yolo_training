import type { CSSProperties, ReactNode } from 'react'

/**
 * 이미지 위에 상자를 겹쳐 그린다.
 *
 * 좌표는 0~1 정규화된 xyxy 다. 데이터셋 검수는 cx/cy/w/h 를 쓰므로 호출부에서 맞춰 넘긴다.
 */
export interface OverlayBox {
  /** [x1, y1, x2, y2], 각각 0~1. */
  box: [number, number, number, number]
  label?: string
  color: string
  dashed?: boolean
  /** 굵게 — 놓친 정답처럼 눈에 먼저 띄어야 하는 것. */
  emphasis?: boolean
}

function boxStyle(b: OverlayBox): CSSProperties {
  const [x1, y1, x2, y2] = b.box
  return {
    display: 'block',
    position: 'absolute',
    left: `${x1 * 100}%`,
    top: `${y1 * 100}%`,
    width: `${Math.max(x2 - x1, 0) * 100}%`,
    height: `${Math.max(y2 - y1, 0) * 100}%`,
    border: `${b.emphasis ? 2.5 : 1.5}px ${b.dashed ? 'dashed' : 'solid'} ${b.color}`,
    borderRadius: 2,
    pointerEvents: 'none',
  }
}

export function BoxOverlay({
  src,
  alt,
  boxes,
  onZoom,
  children,
}: {
  src: string
  alt: string
  boxes: OverlayBox[]
  onZoom?: (url: string) => void
  children?: ReactNode
}) {
  // 상자는 span 이다 — button 안에 div 를 넣으면 콘텐츠 모델을 어긴다.
  const overlay = boxes.map((b, i) => <span key={i} style={boxStyle(b)} title={b.label} />)

  if (!onZoom) {
    return (
      <div style={{ position: 'relative' }}>
        <img className="preview-img" src={src} alt={alt} />
        {overlay}
        {children}
      </div>
    )
  }

  return (
    <button
      className="img-button"
      style={{ position: 'relative' }}
      aria-label={`${alt} 확대`}
      onClick={() => onZoom(src)}
    >
      <img className="preview-img" src={src} alt={alt} />
      {overlay}
      {children}
    </button>
  )
}
