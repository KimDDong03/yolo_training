import type { CSSProperties, ReactNode } from 'react'

/**
 * 이미지 위에 상자를 겹쳐 그린다.
 *
 * 좌표는 0~1 정규화된 xyxy 다. 데이터셋 검수는 cx/cy/w/h 를 쓰므로 호출부에서 맞춰 넘긴다.
 */
/**
 * 상자의 의미. 색과 선 모양은 여기서만 정한다.
 *
 * 예전에는 호출부가 색 문자열을 직접 넘겼다. 그래서 같은 "맞은 검출" 이 화면마다 다른 색이
 * 될 수 있었고, 규칙을 바꾸려면 호출부를 전부 찾아다녀야 했다.
 */
export type BoxKind = 'gt' | 'missed' | 'hit' | 'false' | 'evidence'

const STYLES: Record<BoxKind, { color: string; dashed: boolean; emphasis: boolean }> = {
  gt: { color: '#7fc7a0', dashed: false, emphasis: false }, // 정답
  missed: { color: '#c9a96a', dashed: false, emphasis: true }, // 놓친 정답 — 먼저 눈에 띄어야 한다
  hit: { color: '#8fa9e8', dashed: false, emphasis: false }, // 맞은 검출
  false: { color: '#e27c64', dashed: true, emphasis: false }, // 오검출
  evidence: { color: '#98968f', dashed: true, emphasis: false }, // 근거 박스
}

/**
 * 이미지 위에 상자를 겹쳐 그린다.
 *
 * 좌표는 0~1 정규화된 xyxy 다. 데이터셋 검수는 cx/cy/w/h 를 쓰므로 호출부에서 맞춰 넘긴다.
 */
export interface OverlayBox {
  /** [x1, y1, x2, y2], 각각 0~1. */
  box: [number, number, number, number]
  label?: string
  kind: BoxKind
}

function boxStyle(b: OverlayBox): CSSProperties {
  const [x1, y1, x2, y2] = b.box
  const style = STYLES[b.kind]
  return {
    display: 'block',
    position: 'absolute',
    left: `${x1 * 100}%`,
    top: `${y1 * 100}%`,
    width: `${Math.max(x2 - x1, 0) * 100}%`,
    height: `${Math.max(y2 - y1, 0) * 100}%`,
    border: `${style.emphasis ? 3 : 2}px ${style.dashed ? 'dashed' : 'solid'} ${style.color}`,
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
