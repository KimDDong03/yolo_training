/**
 * 차트 테마.
 *
 * CSS 변수가 아니라 TS 상수인 이유 — Recharts 는 stroke/fill 을 SVG attribute 로 렌더링한다.
 * attribute 값에는 var() 가 해석되지 않으므로 CSS 토큰을 그대로 넘길 수 없다.
 * 다크 전용이라 같은 값이 CSS 쪽에 필요하지도 않다. 값은 styles/tokens.css 와 짝을 이룬다.
 *
 * 축 선과 눈금 글자를 나눠 둔 것은 의도다 — 선은 흐려도 되지만 눈금은 글자라서
 * --panel 위에서 4.5:1 을 넘겨야 한다. 하나로 묶으면 둘 중 하나가 반드시 나빠진다.
 */

/** 지표별 고정 색. run 이 바뀌어도 mAP50 은 항상 같은 색이어야 눈이 헤매지 않는다. */
export const metricSeries: Record<string, string> = {
  mAP50: '#4f8cff', // --accent
  'mAP50-95': '#35c46b', // --ok
  precision: '#e2b23c', // --warn
  recall: '#b07cff',
}

/** 개수가 정해지지 않은 계열(손실 항목, 비교 대상 run)에 순서대로 돌려 쓴다. */
export const palette = [
  '#4f8cff',
  '#35c46b',
  '#e2b23c',
  '#b07cff',
  '#e2564a',
  '#3fc7c7',
]

export function seriesColor(index: number): string {
  return palette[index % palette.length]
}

export const chartAxis = {
  stroke: '#3a4150', // --line-strong
  tick: { fill: '#949cad', fontSize: 11 }, // --muted, 6.3:1
  tickLine: { stroke: '#3a4150' },
}

export const chartGrid = { stroke: '#232833' }

export const chartTooltip = {
  contentStyle: {
    background: '#1d212a', // --panel-2
    border: '1px solid #3a4150',
    borderRadius: 6,
    fontSize: 12,
    color: '#e6e8ee',
  },
  labelStyle: { color: '#949cad' },
}

export const chartLegend = { wrapperStyle: { fontSize: 11 } }
