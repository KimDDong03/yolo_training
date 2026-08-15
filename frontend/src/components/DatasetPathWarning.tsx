import type { Dataset } from '../types'

/**
 * 등록된 원본 폴더를 못 찾을 때 사유를 띄운다.
 *
 * 이 상태에서는 사진만 전부 403/404 가 되고 학습·분석은 멀쩡히 돌아간다. 화면에 아무
 * 말이 없으면 사용자는 원인을 알 방법이 없다. 문장은 서버가 만든 것을 그대로 쓴다.
 */
export function DatasetPathWarning({ dataset }: { dataset: Dataset | null | undefined }) {
  const status = dataset?.path_status
  if (!status || status.ok) return null
  return (
    <div className="help warn" role="status">
      {status.message}
    </div>
  )
}
