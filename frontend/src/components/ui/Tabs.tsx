import type { ReactNode } from 'react'

export interface TabItem {
  key: string
  label: string
}

const tabId = (prefix: string, key: string) => `${prefix}-tab-${key}`
const panelId = (prefix: string, key: string) => `${prefix}-panel-${key}`

/**
 * WAI-ARIA 탭 패턴.
 *
 * 원래 div onClick 이라 키보드로 아예 닿지 않았다. button + 로빙 tabindex 로 바꿔
 * Tab 키는 탭 묶음을 통째로 건너뛰고, 좌우 화살표로 탭 사이를 옮긴다.
 */
export function Tabs({
  items,
  value,
  onChange,
  label,
  idPrefix,
}: {
  items: readonly TabItem[]
  value: string
  onChange: (key: string) => void
  label: string
  idPrefix: string
}) {
  function onKeyDown(e: React.KeyboardEvent<HTMLButtonElement>) {
    const current = items.findIndex((t) => t.key === value)
    let next = -1
    if (e.key === 'ArrowRight') next = (current + 1) % items.length
    else if (e.key === 'ArrowLeft') next = (current - 1 + items.length) % items.length
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = items.length - 1
    if (next < 0) return
    e.preventDefault()
    onChange(items[next].key)
    document.getElementById(tabId(idPrefix, items[next].key))?.focus()
  }

  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          role="tab"
          id={tabId(idPrefix, item.key)}
          className="tab"
          aria-selected={value === item.key}
          aria-controls={panelId(idPrefix, item.key)}
          tabIndex={value === item.key ? 0 : -1}
          onClick={() => onChange(item.key)}
          onKeyDown={onKeyDown}
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}

/** tabIndex=0 은 일부러다 — 패널 안에 포커스 갈 곳이 없어도 키보드로 스크롤할 수 있어야 한다. */
export function TabPanel({
  idPrefix,
  tabKey,
  className,
  children,
}: {
  idPrefix: string
  tabKey: string
  className?: string
  children: ReactNode
}) {
  return (
    <div
      role="tabpanel"
      id={panelId(idPrefix, tabKey)}
      aria-labelledby={tabId(idPrefix, tabKey)}
      tabIndex={0}
      className={className}
    >
      {children}
    </div>
  )
}
