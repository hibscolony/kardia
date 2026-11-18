export type Scheme = 3 | 4;
export type Dataset = {
  version: "kardia-tensors-v1";
  preprocessing: "kardia-v3-notebook";
  source: string;
  rrFormat: "raw";
  samplingRate: 360;
  ecg: number[][][];
  rr: number[][];
  annotations?: string[];
  beatIds?: number[];
};
export type Result = {
  probabilities: number[];
  prediction: number;
  entropy: number;
  disagreement: number;
  elapsed: number;
  scheme: Scheme;
  beat: number;
  source: string;
  timestamp: string;
};
export const labels = ["N", "S", "V", "F"];
export const names = [
  "Normal / bundle branch",
  "Supraventricular ectopic",
  "Ventricular ectopic",
  "Fusion beat",
];
export function validateDataset(v: unknown): Dataset {
  const d = v as Dataset;
  if (
    !d ||
    d.version !== "kardia-tensors-v1" ||
    d.preprocessing !== "kardia-v3-notebook" ||
    d.rrFormat !== "raw" ||
    d.samplingRate !== 360
  )
    throw Error(
      "Use the KARDIA tensor JSON format: notebook-preprocessed ECG, raw RR features, 360 Hz. Raw ECG uploads are not supported.",
    );
  if (typeof d.source !== "string" || !d.source.trim() || d.source.length > 160)
    throw Error("Provide a source name (1–160 characters).");
  if (
    !Array.isArray(d.ecg) ||
    d.ecg.length < 1 ||
    d.ecg.length > 1000 ||
    !Array.isArray(d.rr) ||
    d.rr.length !== d.ecg.length
  )
    throw Error("ECG and RR must contain the same 1–1000 beats.");
  d.ecg.forEach((b, i) => {
    if (
      !Array.isArray(b) ||
      b.length !== 4 ||
      b.some(
        (c) =>
          !Array.isArray(c) ||
          c.length !== 256 ||
          c.some(
            (x) =>
              typeof x !== "number" ||
              !Number.isFinite(x) ||
              Math.abs(x) > 5.00001,
          ),
      )
    )
      throw Error(
        `Beat ${i + 1}: ECG must be [4,256], finite and normalized to [-5,5].`,
      );
    if (
      !Array.isArray(d.rr[i]) ||
      d.rr[i].length !== 8 ||
      d.rr[i].some(
        (x) =>
          typeof x !== "number" || !Number.isFinite(x) || Math.abs(x) > 1e6,
      )
    )
      throw Error(
        `Beat ${i + 1}: RR requires eight finite, unscaled notebook features.`,
      );
  });
  if (
    d.annotations &&
    (!Array.isArray(d.annotations) ||
      d.annotations.length !== d.ecg.length ||
      d.annotations.some((a) => !labels.includes(a)))
  )
    throw Error("Invalid annotation labels.");
  if (
    d.beatIds &&
    (!Array.isArray(d.beatIds) ||
      d.beatIds.length !== d.ecg.length ||
      d.beatIds.some((x) => !Number.isSafeInteger(x) || x < 0))
  )
    throw Error("Invalid beat IDs.");
  return d;
}
export function softmax(a: number[]) {
  const max = Math.max(...a);
  const e = a.map((x) => Math.exp(x - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum);
}
export function ensemble(logits: number[][], bias: number[]) {
  const p = logits.map(softmax);
  return softmax(
    bias.map(
      (b, i) =>
        Math.log(Math.max(p.reduce((s, a) => s + a[i], 0) / p.length, 1e-8)) +
        b,
    ),
  );
}
export function scaleRR(rr: number[], s: { mean: number[]; scale: number[] }) {
  return Float32Array.from(
    rr.map((v, i) =>
      Math.max(
        -8,
        Math.min(
          8,
          Math.fround(Math.fround(Math.fround(v) - s.mean[i]) / s.scale[i]),
        ),
      ),
    ),
  );
}
type Ort = typeof import("onnxruntime-web");
let ort: Ort;
let metadata: any;
let scalers: any;
const sessions = new Map<
  Scheme,
  import("onnxruntime-web").InferenceSession[]
>();
let pending: Promise<void> | null = null;
async function json(path: string) {
  const r = await fetch(import.meta.env.BASE_URL + path);
  if (!r.ok) throw Error(`Could not load ${path} (${r.status})`);
  return r.json();
}
export async function loadModels(
  scheme: Scheme,
  progress: (s: string) => void,
) {
  if (sessions.has(scheme)) {
    progress("Ready · WASM");
    return;
  }
  if (pending) await pending;
  if (sessions.has(scheme)) {
    progress("Ready · WASM");
    return;
  }
  pending = (async () => {
    progress("Initializing browser runtime");
    ort = await import("onnxruntime-web/wasm");
    ort.env.wasm.numThreads = 1;
    const { default: wasmModuleUrl } = await import(
      "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url"
    );
    ort.env.wasm.wasmPaths = {
      wasm: new URL(
        import.meta.env.BASE_URL + "runtime/ort-wasm-simd-threaded.wasm",
        window.location.href,
      ).href,
      mjs: new URL(wasmModuleUrl, window.location.href).href,
    };
    [metadata, scalers] = await Promise.all([
      json("models/model_metadata.json"),
      json("models/scalers.json"),
    ]);
    const created: import("onnxruntime-web").InferenceSession[] = [];
    try {
      for (const seed of [42, 123]) {
        progress(`Loading seed ${seed} · ${created.length + 1}/2`);
        created.push(
          await ort.InferenceSession.create(
            import.meta.env.BASE_URL +
              `models/kardia_${scheme}class_seed${seed}.onnx`,
            { executionProviders: ["wasm"], graphOptimizationLevel: "all" },
          ),
        );
      }
      sessions.set(scheme, created);
    } catch (e) {
      for (const s of created) await s.release();
      throw e;
    }
    progress("Ready · WASM");
  })();
  try {
    await pending;
  } finally {
    pending = null;
  }
}
export async function infer(
  d: Dataset,
  beat: number,
  scheme: Scheme,
): Promise<Result> {
  const models = sessions.get(scheme);
  if (!models) throw Error("AI Model Not Connected");
  if (d.annotations?.[beat] === "F" && scheme === 3)
    throw Error(
      "Fusion (F) is excluded from the three-class experiment. Select four classes.",
    );
  const start = performance.now();
  const ecg = new ort.Tensor(
    "float32",
    Float32Array.from(d.ecg[beat].flat()),
    [1, 4, 256],
  );
  const rr = new ort.Tensor(
    "float32",
    scaleRR(d.rr[beat], scalers[String(scheme)]),
    [1, 8],
  );
  const logits: number[][] = [];
  try {
    for (const m of models) {
      const output = await m.run({ ecg, rr });
      try {
        const a = Array.from(output[m.outputNames[0]].data as Float32Array);
        if (a.length !== scheme || a.some((x) => !Number.isFinite(x)))
          throw Error("Invalid model output");
        logits.push(a);
      } finally {
        Object.values(output).forEach((t) => t.dispose());
      }
    }
  } finally {
    ecg.dispose();
    rr.dispose();
  }
  const probabilities = ensemble(
    logits,
    metadata.schemes[String(scheme)].logit_bias,
  );
  const seedP = logits.map(softmax);
  return {
    probabilities,
    prediction: probabilities.indexOf(Math.max(...probabilities)),
    entropy:
      -probabilities.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0) /
      Math.log(scheme),
    disagreement:
      seedP[0].reduce((s, p, i) => s + Math.abs(p - seedP[1][i]), 0) / 2,
    elapsed: performance.now() - start,
    scheme,
    beat,
    source: d.source,
    timestamp: new Date().toISOString(),
  };
}
