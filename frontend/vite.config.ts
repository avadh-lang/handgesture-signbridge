import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // dev-only: the built app is served by backend/server.py itself, but
      // `npm run dev` runs on its own port - proxy the WebSocket through so
      // the frontend can always just connect to its own origin.
      '/ws': {
        target: 'ws://127.0.0.1:8765',
        ws: true,
      },
    },
  },
})
