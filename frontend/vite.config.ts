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
  },
})
