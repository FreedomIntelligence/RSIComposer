import { attributions } from './attributions.mjs';
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import AutoImport from 'unplugin-auto-import/vite';
import json5 from 'vite-plugin-json5';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  define: {'process.env.NODE_ENV': JSON.stringify('production'), __INTLIFY_JIT_COMPILATION__: true},
  plugins: [attributions(), vue(), json5(), AutoImport({imports: ['vue'], dts: false})],
  resolve: {alias: {'@swanlab-vue': fileURLToPath(new URL('./swanboard', import.meta.url))}},
  build: {
    outDir: '../static/app', emptyOutDir: true,
    lib: {entry: 'main.js', formats: ['iife'], name: 'RSIMonitorApp', fileName: () => 'monitor.js', cssFileName: 'monitor'},
    rollupOptions: {output: {inlineDynamicImports: true}},
  },
});
