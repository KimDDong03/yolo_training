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
  mAP50: '#e8c87c', // --accent
  'mAP50-95': '#7fc7a0', // --ok
  precision: '#8fa9e8',
  recall: '#c9a96a', // --warn
}

/** 개수가 정해지지 않은 계열(손실 항목, 비교 대상 run)에 순서대로 돌려 쓴다. */
export const palette = [
  '#e8c87c',
  '#7fc7a0',
  '#8fa9e8',
  '#e27c64',
  '#c9a96a',
  '#9a8fd8',
]

export function seriesColor(index: number): string {
  return palette[index % palette.length]
}

export const chartAxis = {
  stroke: '#43464e', // --line-strong
  tick: { fill: '#98968f', fontSize: 10 }, // --muted-2, 5.31:1
  tickLine: { stroke: '#43464e' },
}

export const chartGrid = { stroke: '#2c2e35' } // --panel-2

export const chartTooltip = {
  contentStyle: {
    background: '#2c2e35', // --panel-2
    border: '1px solid #43464e',
    borderRadius: 9,
    fontSize: 12,
    color: '#f5f3ee',
  },
  labelStyle: { color: '#98968f' },
}

export const chartLegend = { wrapperStyle: { fontSize: 11 } }
