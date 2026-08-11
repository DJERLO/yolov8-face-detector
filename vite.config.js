import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs",
                    dest: ".",
                },
                {
                    src: "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm",
                    dest: ".",
                },
            ],
        }),
    ],
});