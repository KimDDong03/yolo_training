import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 단독망 배포를 전제로 한다: 외부 CDN 을 쓰지 않고 모든 의존성을 번들에 인라인한다.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    rollupOptions: {
      output: {
        // 한 덩어리 605KB 라 Vite 가 매 빌드마다 경고했다. 새 의존성 없이 내장 기능으로
        // 나눈다 — 인라인 방침(위 주석)과 무관하다. 청크를 나누는 것과 CDN 을 쓰는 것은
        // 다른 얘기다. react 는 앱 코드와 달리 거의 안 바뀌어 브라우저 캐시가 오래 산다.
        manualChunks: {
          vendor: ['react', 'react-dom'],
        },
      },
    },
  },
})
