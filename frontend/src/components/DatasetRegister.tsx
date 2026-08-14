import { useId, useRef, useState } from 'react'
import { api } from '../api'
import { useToast } from './ui/Toast'

/**
 * zip 업로드 · 서버 경로 등록.
 *
 * NewRunPanel 안에 있던 것을 꺼냈다 — 데이터셋 관리 화면에서도 같은 UI 가 필요하다.
 */
export function DatasetRegister({ onDone }: { onDone: () => void }) {
  const [path, setPath] = useState('')
  const [valRatio, setValRatio] = useState(0.2)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [over, setOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const pathId = useId()
  const ratioId = useId()
  const toast = useToast()

  /** 성공하면 true. 실패 메시지는 인라인으로 남긴다 — 다음 행동을 정하는 정보라 사라지면 안 된다. */
  async function run(work: () => Promise<unknown>, done: string) {
    setBusy(true)
    setError('')
    try {
      await work()
      toast(done, 'success')
      onDone()
      return true
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
      return false
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h3>데이터셋 등록</h3>

      <div
        className={`drop ${over ? 'over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          const file = e.dataTransfer.files[0]
          if (file) run(() => api.uploadZip(file, file.name.replace(/\.zip$/i, ''), valRatio), `${file.name} 등록됨`)
        }}
      >
        {/*
          클릭 대상은 버튼이어야 키보드로 닿는다. 파일 input 을 이 안에 두면
          대화형 요소가 중첩되므로 형제로 뺀다.
        */}
        <button type="button" disabled={busy} onClick={() => fileRef.current?.click()}>
          {busy ? '처리 중…' : 'zip 파일을 여기에 끌어다 놓거나 클릭해서 선택'}
        </button>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept=".zip"
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) run(() => api.uploadZip(file, file.name.replace(/\.zip$/i, ''), valRatio), `${file.name} 등록됨`)
        }}
      />

      <div className="row" style={{ marginTop: 10 }}>
        <div className="field" style={{ flex: 1 }}>
          <label className="sr-only" htmlFor={pathId}>
            서버 폴더 경로
          </label>
          <input
            id={pathId}
            placeholder="또는 서버 폴더 경로 지정 — 예: D:\data\coins"
            value={path}
            onChange={(e) => setPath(e.target.value)}
          />
        </div>
        <button
          className="nowrap"
          disabled={!path || busy}
          onClick={async () => {
            const ok = await run(
              () => api.registerPath(path, path.split(/[\\/]/).filter(Boolean).pop() ?? path, valRatio),
              '경로가 등록되었습니다',
            )
            if (ok) setPath('')
          }}
        >
          경로 등록
        </button>
      </div>

      <div className="row small muted wrap" style={{ marginTop: 8 }}>
        <label htmlFor={ratioId}>train/val 이 없을 때 검증 비율</label>
        <input
          id={ratioId}
          type="number"
          step={0.05}
          min={0.05}
          max={0.5}
          value={valRatio}
          onChange={(e) => setValRatio(parseFloat(e.target.value))}
          style={{ width: 80 }}
        />
        <span>경로 등록은 파일을 복사하지 않고 원본을 그대로 참조합니다.</span>
      </div>

      {error && (
        <div className="error small" style={{ marginTop: 8 }}>
          {error}
        </div>
      )}
    </div>
  )
}
