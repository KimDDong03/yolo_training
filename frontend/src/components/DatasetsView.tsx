import { api } from '../api'
import type { Dataset } from '../types'
import type { LoadStatus } from '../useResource'
import { DatasetRegister } from './DatasetRegister'
import { useConfirm } from './ui/Dialog'
import { EmptyState, SkeletonRows } from './ui/EmptyState'
import { useToast } from './ui/Toast'

interface Props {
  datasets: Dataset[]
  status: LoadStatus
  onChanged: () => void
}

/** 데이터셋 등록과 목록을 한 화면에 모은다. 예전에는 등록은 학습 설정에, 목록은 오른쪽 아래에 있었다. */
export function DatasetsView({ datasets, status, onChanged }: Props) {
  const confirm = useConfirm()
  const toast = useToast()

  async function remove(dataset: Dataset) {
    const ok = await confirm({
      title: `'${dataset.name}' 을 삭제할까요?`,
      body:
        dataset.source === 'zip'
          ? '업로드된 사본이 지워집니다. 이 데이터셋으로 학습한 기록은 남습니다.'
          : '등록만 해제합니다. 원본 폴더는 건드리지 않습니다.',
      confirmLabel: '삭제',
      danger: true,
    })
    if (!ok) return
    try {
      await api.deleteDataset(dataset.id)
      toast(`${dataset.name} 삭제됨`, 'success')
    } catch (e) {
      toast(String(e instanceof Error ? e.message : e), 'error')
    }
    onChanged()
  }

  return (
    <div className="pane">
      <DatasetRegister onDone={onChanged} />

      <div className="card">
        <h3>등록된 데이터셋</h3>

        {status === 'loading' && <SkeletonRows rows={3} />}

        {status === 'error' && (
          <EmptyState
            tone="error"
            title="데이터셋을 불러오지 못했습니다"
            description="서버가 응답하지 않습니다."
            action={
              <button className="btn-sm" onClick={onChanged}>
                다시 시도
              </button>
            }
          />
        )}

        {status === 'ready' && datasets.length === 0 && (
          <EmptyState title="등록된 데이터셋이 없습니다" description="위에서 zip 을 올리거나 서버 폴더 경로를 지정하세요." />
        )}

        {datasets.length > 0 && (
          <table>
            <caption className="sr-only">등록된 데이터셋 목록</caption>
            <thead>
              <tr>
                <th scope="col">이름</th>
                <th scope="col">출처</th>
                <th scope="col">이미지</th>
                <th scope="col">클래스</th>
                <th scope="col">
                  <span className="sr-only">작업</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.id}>
                  <td>{d.name}</td>
                  <td className="muted">{d.source === 'zip' ? 'zip 업로드' : '경로 참조'}</td>
                  <td>{d.report.total_images.toLocaleString()}</td>
                  <td className="muted">{d.classes.join(', ')}</td>
                  <td>
                    <button className="btn-xs" onClick={() => remove(d)}>
                      삭제
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
