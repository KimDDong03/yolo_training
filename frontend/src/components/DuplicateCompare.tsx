import { useEffect, useState, type CSSProperties } from 'react'

import { api } from '../api'
import type { ImageInfo } from '../types'
import { Modal } from './ui/Dialog'

/** 깜빡임 주기. 이보다 빠르면 잔상이 남아 차이가 아니라 흔들림으로 보인다. */
const BLINK_MS = 450
/** 사진 상자의 최대 높이(vh). CSS 의 max-height 와 반드시 같은 값이어야 한다. */
const BOX_MAX_VH = 58

export interface CompareImage {
  path: string
  split: 'train' | 'val'
  /** 이 장에만 붙는 한 줄 (누수 쌍의 유사도 같은 것). */
  note?: string
}

function name(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

function bytes(size: number): string {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

/**
 * 중복 묶음 비교.
 *
 * 썸네일을 나란히 놓는 것만으로는 "같은 사진인가" 가 갈리지 않는다. 인접한 MRI 슬라이스처럼
 * 눈으로 훑으면 똑같아 보이는 것이 실제로는 다른 사진이다. 그래서 두 가지를 준다 —
 * 겹쳐 놓고 가르거나 깜빡여 **차이 나는 자리를 눈에 띄게** 만드는 것, 그리고 해상도·용량·
 * 정답 박스 수처럼 **어느 쪽을 남길지 정하는 숫자**.
 */
export function DuplicateCompare({
  datasetId,
  title,
  subtitle,
  images,
  marked,
  keeperPath,
  onToggle,
  onKeepOnly,
  onClose,
  onStep,
}: {
  datasetId: string
  title: string
  subtitle?: string
  images: CompareImage[]
  /** 지우기로 고른 경로들. */
  marked: Set<string>
  /** 이 묶음에서 남기기로 정해진 한 장. 없으면 null — 판정은 호출부가 한다. */
  keeperPath: string | null
  onToggle: (path: string) => void
  /** 이 한 장만 남기고 묶음의 나머지를 전부 고른다 — 화면에 안 보이는 장까지. */
  onKeepOnly: (path: string) => void
  onClose: () => void
  /** 묶음 사이 이동. 없으면 화살표를 그리지 않는다. */
  onStep?: (delta: number) => void
}) {
  const [other, setOther] = useState(1)
  const [overlay, setOverlay] = useState(false)
  const [wipe, setWipe] = useState(50)
  const [blink, setBlink] = useState(false)
  const [showTop, setShowTop] = useState(true)
  const [info, setInfo] = useState<Record<string, ImageInfo>>({})

  const base = images[0]
  const comparedIndex = Math.min(other, images.length - 1)
  const compared = images[comparedIndex]
  // 번호는 묶음 안의 실제 자리다. 늘 1번/2번으로 적으면 3장 이상 묶음에서 위 선택기의
  // "3번" 과 아래 표의 "2번" 이 서로 다른 말을 하게 된다.
  const shown = [
    { image: base, number: 1 },
    { image: compared, number: comparedIndex + 1 },
  ]

  // 다른 묶음으로 넘어가면 보기 상태를 처음으로 되돌린다. 안 그러면 3장짜리에서 2번을 보다가
  // 2장짜리로 넘어갔을 때 빈 자리를 보게 된다.
  useEffect(() => {
    setOther(1)
    setWipe(50)
  }, [base?.path])

  // 열려 있는 묶음의 것만 받는다. 리포트 전체를 미리 받으면 30묶음 × N장이 된다.
  useEffect(() => {
    let cancelled = false
    for (const image of images) {
      if (info[image.path]) continue
      api
        .datasetImageInfo(datasetId, image.path)
        .then((got) => !cancelled && setInfo((prev) => ({ ...prev, [image.path]: got })))
        .catch(() => {
          /* 못 읽으면 숫자 줄만 비운다. 비교 자체는 계속된다. */
        })
    }
    return () => {
      cancelled = true
    }
  }, [datasetId, images, info])

  useEffect(() => {
    if (!blink) return
    const timer = setInterval(() => setShowTop((v) => !v), BLINK_MS)
    return () => clearInterval(timer)
  }, [blink])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // 슬라이더에 포커스가 있으면 화살표는 슬라이더 것이다.
      const tag = (event.target as HTMLElement | null)?.tagName
      if (event.key === 'b' || event.key === 'B') {
        setOverlay(true)
        setBlink((v) => !v)
      } else if (event.key === 'ArrowLeft' && onStep && tag !== 'INPUT') {
        onStep(-1)
      } else if (event.key === 'ArrowRight' && onStep && tag !== 'INPUT') {
        onStep(1)
      } else {
        return
      }
      event.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onStep])

  if (!base || !compared) return null

  // 상자는 기준 장의 비율로 고정한다. 높이만 막으면 세로 사진에서 폭이 그대로 남아
  // 상자가 가로로 납작해지고 사진은 그 안에 작게 박힌다 — 폭도 함께 좁혀야 한다.
  const baseInfo = info[base.path]
  const boxStyle =
    baseInfo?.width && baseInfo?.height
      ? {
          aspectRatio: `${baseInfo.width} / ${baseInfo.height}`,
          maxWidth: `calc(${BOX_MAX_VH}vh * ${baseInfo.width / baseInfo.height})`,
        }
      : { aspectRatio: '4 / 3' }

  return (
    <Modal open onClose={onClose} className="dialog compare" labelledBy="compare-title">
      <div className="compare-head">
        <div>
          <h3 id="compare-title" style={{ margin: 0 }}>
            {title}
          </h3>
          {subtitle && <p className="small muted" style={{ margin: '2px 0 0' }}>{subtitle}</p>}
        </div>
        <div className="row" style={{ gap: 6, marginLeft: 'auto' }}>
          {onStep && (
            <>
              <button className="btn-sm" onClick={() => onStep(-1)} aria-label="이전 묶음">
                ←
              </button>
              <button className="btn-sm" onClick={() => onStep(1)} aria-label="다음 묶음">
                →
              </button>
            </>
          )}
          <button className="btn-sm" data-autofocus onClick={onClose}>
            닫기
          </button>
        </div>
      </div>

      <div className="row small" style={{ gap: 6, marginTop: 'var(--sp-3)' }}>
        <button
          className={overlay ? 'btn-sm' : 'btn-sm primary'}
          onClick={() => {
            setOverlay(false)
            setBlink(false)
          }}
        >
          나란히
        </button>
        <button
          className={overlay ? 'btn-sm primary' : 'btn-sm'}
          onClick={() => setOverlay(true)}
        >
          겹쳐 보기
        </button>
        {overlay && (
          <>
            <button className="btn-sm" onClick={() => setBlink((v) => !v)}>
              {blink ? '깜빡임 끄기' : '깜빡임 (B)'}
            </button>
            {!blink && (
              <label className="row small" style={{ gap: 6, marginLeft: 6 }}>
                <span className="muted">가르기</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={wipe}
                  onChange={(e) => setWipe(Number(e.target.value))}
                  style={{ width: 160 }}
                  aria-label="겹친 사진 가르기"
                />
              </label>
            )}
          </>
        )}
        {images.length > 2 && (
          <label className="row small" style={{ gap: 6, marginLeft: 'auto' }}>
            <span className="muted">비교 대상</span>
            <select
              value={Math.min(other, images.length - 1)}
              onChange={(e) => setOther(Number(e.target.value))}
            >
              {images.slice(1).map((image, index) => (
                <option key={image.path} value={index + 1}>
                  {index + 2}번 · {name(image.path)}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {overlay ? (
        <div className="compare-stack" style={boxStyle}>
          <img src={api.datasetImageUrl(datasetId, base.path)} alt={name(base.path)} />
          <img
            src={api.datasetImageUrl(datasetId, compared.path)}
            alt={name(compared.path)}
            style={
              blink
                ? { opacity: showTop ? 1 : 0 }
                : { clipPath: `inset(0 ${100 - wipe}% 0 0)` }
            }
          />
          <span className="compare-stack-cap">
            {blink
              ? `깜빡이는 중 — 지금 ${showTop ? '2번' : '1번'}`
              : `왼쪽 ${name(compared.path)} · 오른쪽 ${name(base.path)}`}
          </span>
        </div>
      ) : (
        <div className="compare-panes" style={{ marginTop: 'var(--sp-3)' }}>
          {shown.map(({ image, number }, index) => (
            <Pane
              key={image.path}
              datasetId={datasetId}
              image={image}
              number={number}
              boxStyle={boxStyle}
              info={info[image.path]}
              rival={info[(index === 0 ? compared : base).path]}
              marked={marked.has(image.path)}
              keeper={keeperPath === image.path}
              onToggle={() => onToggle(image.path)}
              onKeepOnly={() => onKeepOnly(image.path)}
            />
          ))}
        </div>
      )}

      {overlay && (
        <div className="compare-panes" style={{ marginTop: 'var(--sp-3)' }}>
          {shown.map(({ image, number }, index) => (
            <Facts
              key={image.path}
              image={image}
              number={number}
              info={info[image.path]}
              rival={info[(index === 0 ? compared : base).path]}
              marked={marked.has(image.path)}
              keeper={keeperPath === image.path}
              onToggle={() => onToggle(image.path)}
              onKeepOnly={() => onKeepOnly(image.path)}
            />
          ))}
        </div>
      )}
    </Modal>
  )
}

function Pane({
  datasetId,
  image,
  number,
  boxStyle,
  info,
  rival,
  marked,
  keeper,
  onToggle,
  onKeepOnly,
}: {
  datasetId: string
  image: CompareImage
  number: number
  boxStyle: CSSProperties
  info?: ImageInfo
  rival?: ImageInfo
  marked: boolean
  keeper: boolean
  onToggle: () => void
  onKeepOnly: () => void
}) {
  return (
    <div className="compare-pane">
      <div className="compare-box" style={boxStyle}>
        <img src={api.datasetImageUrl(datasetId, image.path)} alt={name(image.path)} />
      </div>
      <Facts
        image={image}
        number={number}
        info={info}
        rival={rival}
        marked={marked}
        keeper={keeper}
        onToggle={onToggle}
        onKeepOnly={onKeepOnly}
      />
    </div>
  )
}

/** 어느 쪽을 남길지 정하는 숫자. 상대보다 큰 값에 표시를 남긴다. */
function Facts({
  image,
  number,
  info,
  rival,
  marked,
  keeper,
  onToggle,
  onKeepOnly,
}: {
  image: CompareImage
  number: number
  info?: ImageInfo
  rival?: ImageInfo
  marked: boolean
  keeper: boolean
  onToggle: () => void
  onKeepOnly: () => void
}) {
  const wins = (mine?: number | null, theirs?: number | null) =>
    mine != null && theirs != null && mine > theirs ? 'compare-better' : undefined
  const pixels = (got?: ImageInfo) =>
    got?.width && got?.height ? got.width * got.height : null

  return (
    <div className="compare-facts">
      <div className="row small" style={{ gap: 6 }}>
        <span className="badge">{number}번</span>
        <span className="badge">{image.split === 'train' ? '학습' : '검증'}</span>
        <span className="mono" title={image.path}>
          {name(image.path)}
        </span>
        {/* 목록에는 "이것만 남기기" 하나뿐이다. 정밀하게 고르는 자리는 여기다 —
            묶음 단위로 정하는 버튼과 이 한 장만 뒤집는 토글을 함께 둔다. */}
        <button
          className="btn-sm spacer"
          onClick={onKeepOnly}
          aria-pressed={keeper}
        >
          {keeper ? '남길 장' : '이것만 남기기'}
        </button>
        <button
          className={marked ? 'btn-sm danger' : 'btn-sm'}
          onClick={onToggle}
          aria-pressed={marked}
        >
          {marked ? '지움' : '남김'}
        </button>
      </div>
      <div className="row small compare-nums" style={{ gap: 10, marginTop: 4 }}>
        {info ? (
          <>
            <span className={wins(pixels(info), pixels(rival))}>
              {info.width && info.height ? `${info.width}×${info.height}` : '크기 모름'}
            </span>
            <span className={wins(info.bytes, rival?.bytes)}>{bytes(info.bytes)}</span>
            <span className={wins(info.boxes, rival?.boxes)}>정답 박스 {info.boxes}개</span>
          </>
        ) : (
          <span className="muted">읽는 중…</span>
        )}
        {image.note && <span className="muted">{image.note}</span>}
      </div>
    </div>
  )
}
