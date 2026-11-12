import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const httpTarget = (env.VITE_DEV_BACKEND_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const wsTarget = httpTarget.replace(/^http/, 'ws');

  return {
    plugins: [react()],
    server: {
      host: '127.0.0.1',
      proxy: {
        '/api/health': {
          target: httpTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/health/, '/health'),
        },
        '/ws/transcribe': {
          target: wsTarget,
          ws: true,
          changeOrigin: true,
        },
      },
    },
  };
});
