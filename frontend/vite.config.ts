import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// The dev server proxies API paths to the FastAPI backend, so the frontend
// needs no CORS configuration and calls the same paths it would in production.
const target = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/chat": target,
      "/history": target,
      "/conversations": target,
      "/health": target,
    },
  },
});
