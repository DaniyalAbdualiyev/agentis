import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server listens on 3000 to match the docker-compose port mapping.
// In production the app is built to static files and served by nginx.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
  },
});
