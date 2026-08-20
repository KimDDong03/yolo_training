import { useState } from 'react'
import { api } from '../api'
import type { Dataset } from '../types'
import type { LoadStatus } from '../useResource'
import { DatasetRegister } from './DatasetRegister'
import { DatasetReviewPanel } from './DatasetReviewPanel'
import { QualityPanel } from './QualityPanel'
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
  const [openReview, setOpenReview] = useState<string | null>(null)

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
      <div className="new-run-body">
        <DatasetRegister onDone={onChanged} />

        <h3 style={{ margin: '38px 0 var(--sp-4)', fontSize: 'var(--fs-xl)', letterSpacing: '-0.02em' }}>
          등록된 데이터셋
        </h3>

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
          <EmptyState
            title="등록된 데이터셋이 없습니다"
            description="위에서 zip 을 올리거나 서버 폴더 경로를 지정하세요."
          />
        )}

        {/*
          표를 카드로 바꿨다. 표는 열 하나에 값 하나씩만 담을 수 있어서 정작 판단에 필요한
          train/val 비율과 검수 경고가 들어갈 자리가 없었다.
        */}
        <div className="ds-list">
          {datasets.map((d) => {
            const issues = Object.values(d.report.issue_counts ?? {}).reduce((a, b) => a + b, 0)
            const open = openReview === d.id
            return (
              <div className="ds-card" key={d.id}>
                <div className="ds-head">
                  <span className="ds-name">{d.name}</span>
                  <span className="badge">{d.source === 'zip' ? 'zip 업로드' : '경로 참조'}</span>
                  <button
                    className="btn-sm spacer"
                    aria-expanded={open}
                    onClick={() => setOpenReview(open ? null : d.id)}
                  >
                    {open ? '검수 접기' : '검수 보기'}
                  </button>
                  <button className="btn-sm danger" onClick={() => remove(d)}>
                    삭제
                  </button>
                </div>

                <div className="summary-line" style={{ marginTop: 'var(--sp-3)' }}>
                  <span>이미지 {d.report.total_images.toLocaleString()}</span>
                  {d.report.train_count != null && d.report.val_count != null && (
                    <span>
                      train/val {d.report.train_count.toLocaleString()} / {d.report.val_count.toLocaleString()}
                    </span>
                  )}
                  <span>클래스 {d.classes.length}</span>
                  {issues > 0 && <span className="warn">검수 경고 {issues}건</span>}
                </div>

                <div className="ds-path">{d.root}</div>

                {open && (
                  <div style={{ marginTop: 'var(--sp-4)' }}>
                    <DatasetReviewPanel dataset={d} />
                    <QualityPanel dataset={d} />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
