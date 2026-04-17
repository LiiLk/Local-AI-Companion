import { resolve } from "node:path";
import { defineConfig, normalizePath } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: normalizePath(resolve(__dirname, "../../frontend/live2d/live2d.js")),
          dest: "live2d",
        },
        {
          src: normalizePath(resolve(__dirname, "../../frontend/live2d/runtime-assets")),
          dest: ".",
        },
      ],
    }),
  ],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  preview: {
    port: 4173,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
