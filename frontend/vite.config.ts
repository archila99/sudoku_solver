import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/solve": {
        target: process.env.VITE_API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/upload": {
        target: process.env.VITE_API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
