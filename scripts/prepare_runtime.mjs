import {mkdir,copyFile} from 'node:fs/promises';
await mkdir('public/runtime',{recursive:true});
for(const ext of ['wasm','mjs']) await copyFile(`node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.${ext}`,`public/runtime/ort-wasm-simd-threaded.${ext}`);
