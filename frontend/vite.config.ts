import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev only: proxy API calls to the FastAPI server on :8000.
// In production the same FastAPI process serves the built frontend,
// so the app calls same-origin paths and needs no proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://localhost:8000",
      "/drugs": "http://localhost:8000",
      "/ask": "http://localhost:8000",
      "/ask-agent": "http://localhost:8000",
      "/index": "http://localhost:8000",
    },
  },
});
