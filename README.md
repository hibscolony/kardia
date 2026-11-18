# KARDIA — Intelligent ECG Analysis

Research ECG workspace built with React, TypeScript, Vite, Tailwind CSS, Lucide, Recharts, Framer Motion and ONNX Runtime Web. All inference runs locally using the supplied original ONNX models. No Python server, backend, random predictions or simulated model outputs.

## Run locally

Requires Node.js 22+ and npm.

```sh
npm ci
npm run dev
```

Open the localhost URL printed by Vite. Go to **ECG Analyzer → Load MIT-BIH demo → Analyze heartbeat**. Switch between four-class and three-class ensembles. Each model loads asynchronously on demand; first use downloads approximately 6.4 MB of models plus the WASM runtime. Uploads are processed in memory and are not sent to a server or persisted.

```sh
npm test
npm run build
npm run preview
```

`dist/` is the static deployment output. Do not open `index.html` using `file://`: fetch and WebAssembly require an HTTP(S) origin. GitHub Pages provides HTTPS. A SIMD-capable modern browser is required. A single WASM thread avoids cross-origin-isolation requirements. WebGPU is deliberately not enabled.

## GitHub Pages

1. Put the source in a GitHub repository on branch `main` (include `public/models/`, `public/data/`, package-lock.json and `.github/workflows/deploy.yml`).
2. In repository **Settings → Pages**, select **GitHub Actions** as the build source.
3. Push to `main` or open **Actions → Deploy KARDIA to GitHub Pages → Run workflow** to deploy.
4. The workflow runs unit tests, type-checks/builds, uploads `dist/`, and deploys to Pages. The public URL is https://hibscolony.github.io/kardia/ after successful deployment.

Vite uses a relative base and navigation uses URL hashes, so repository subpaths and refreshes work without rewrite rules. `npm ci` restores the exact pinned dependency tree and copies the matching runtime files automatically. The build emits the WASM module loader as a bundled asset. No secret keys are needed.

Repository: https://github.com/hibscolony/kardia (public). Tests, builds and Pages deployment run on pushes. GitHub Pages must first be enabled in the repository's Pages settings. If it is not yet enabled, the deployment job may fail until you configure the source and rerun the workflow.

## Five application pages

- Overview: real MIT-BIH waveform and both research experiments.
- ECG Analyzer: real demo, validated JSON upload, beat/channel navigation, original ensemble inference, probabilities, entropy, seed disagreement, PDF export.
- Model Performance: accuracy, macro/weighted F1, class F1, original confusion matrices and test-set distribution.
- How KARDIA Works: interactive seven-stage architecture explorer.
- Research & Docs: provenance, exact preprocessing, dataset protocol, limitations and original metadata download.

## Input contract

Only **notebook-preprocessed ECG tensors** with **unscaled RR features** are accepted. Arbitrary raw ECG, images, PDF, CSV and automatic R-peak detection are not supported or validated.

Download `public/data/demo.json` from the input-format section for a complete valid example. The contract is:

```text
version: "kardia-tensors-v1"
preprocessing: "kardia-v3-notebook"
source: string, 1–160 characters
samplingRate: 360
rrFormat: "raw"
ecg: number[beats][4][256]     (normalized, finite, bounded to ±5)
rr: number[beats][8]           (raw notebook RR features, NOT standardized)
annotations?: ("N" | "S" | "V" | "F")[beats]
beatIds?: nonnegative integer[beats]
```

Limits: 1–1000 beats, 30 MB per file. Upload validation checks shapes, finite values, metadata and ranges; it cannot prove that a supplied waveform was clinically valid or processed correctly. Never relabel arbitrary arrays to bypass the contract.

Exact notebook preprocessing:

1. Select MLII as lead 1 if available, otherwise the first channel; select the other lead as lead 2. Original records have 360 Hz sampling.
2. Interpolate non-finite samples as in the notebook; filter the **full record** with SciPy `butter(3, [0.5,40], btype="bandpass", fs=360, output="sos")` and `sosfiltfilt(..., axis=0)`, cast to float32. Do not substitute causal filtering or filter each cropped beat separately.
3. Use MIT-BIH expert beat annotations, including `/`, `f` and `Q` for RR history but not as classification targets. For valid AAMI N/S/V/F targets, crop `[peak-112:peak+144]` for current and previous beats. Invalid previous windows fall back to the current beat.
4. For each lead in each window: subtract the median, divide by `max(percentile(abs(window),95),0.05)`, clip to [-5,5], cast float32. Concatenate `[current_lead1,current_lead2,previous_lead1,previous_lead2]`.
5. Compute up to eight preceding RR intervals in seconds. Feature order: previous RR, second previous RR, previous/second (denominator minimum 1e-3), previous/median (minimum 1e-3), median, mean, population standard deviation / mean (minimum 1e-3), previous minus second. The notebook uses 0.8 s for an empty history and previous RR when second previous is missing.
6. The browser converts raw RR values to float32, subtracts the selected training scaler mean, divides by its scale, rounds each operation as float32 (matching StandardScaler), and clips to [-8,8]. Do not standardize uploaded RR features beforehand.

## Original-model inference

`src/inference.ts` loads the original `kardia_{3|4}class_seed{42|123}.onnx` models with WASM. Every request runs batch size 1 with inputs `ecg [1,4,256]`, `rr [1,8]`.

For each seed compute stable `softmax(logits)`. Average seed probabilities, then compute `softmax(log(max(mean_probs,1e-8)) + logit_bias)` using the original metadata. Four-class bias = [0,1.2,0.5,-0.6]; three-class bias = [0,0,0.5]. No inference fixture is used by the production app.

Uncertainty displays normalized Shannon entropy of the final distribution and total variation between the two seed distributions before bias. These are descriptive quantities, not calibrated clinical risk or conformal coverage guarantees. Known F demo/upload annotations cannot be analyzed with the three-class model. Unlabeled unknown F beats cannot be detected as out-of-scope by that model.

## Research sources and scope

- Original models/scalers/metadata: user-supplied `kardia_web_onnx.zip`.
- Method: `notebook9963638c15.ipynb`, extracted verbatim to `research_notebook.py` for auditable function reuse. The entire training notebook is **not** executed by the web app or preparation script.
- Evaluation: matching `KARDIA_V3_Dual_Scheme_Outputs.zip` in the user's Downloads. Imported metrics match ONNX metadata. Confusion matrices are the original unmodified PNGs, not inferred from rounded scores.
- Demo: 12 actual beats from MIT-BIH record 100, selected to include N/S/V annotations. Record 100 contains no F demo beat. Eight consecutive ordinary beats plus available rare-class examples are selected; the displayed demo sequence is **not** a continuous rhythm strip. `beatIds` retain original extracted-beat indices. The overview strip is a separate continuous 8-second raw MLII excerpt starting at sample 3600.
- Dataset: [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/), Moody & Mark (2005), DOI [10.13026/C2F305](https://doi.org/10.13026/C2F305). Data license: Open Data Commons Attribution License v1.0. Preserve attribution when redistributing demo data.
- The supplied research artifacts do not identify team names or affiliation; these are not invented. Model/source redistribution rights remain with their respective owners.

Four-class: accuracy 93.1649456%, macro F1 0.63103847, weighted F1 0.93055687, 47,549 test beats. Three-class: accuracy 95.2631356%, macro F1 0.74771459, weighted F1 0.947858 (rounded), 47,162 test beats. Fusion is excluded from the three-class experiment; the scopes are not interchangeable.

Protocol: DS1 split into 16 training, 3 validation (223/207/205), 3 calibration (124/118/106) records; test = 21 DS2 records with 202 excluded to avoid same-patient overlap with 201. Record 208 stays in training. Test data is not used for model selection or bias tuning.

## Reproduce demo and verify

The web app needs no Python. Optional **offline preparation only** requires Python packages `numpy scipy wfdb onnxruntime`; use an isolated virtual environment. Run `python scripts/prepare_demo.py` to download record 100 (or reuse it in `artifacts/mitdb`), execute only the exact notebook `extract_record` function, export demo JSON/scalers and regenerate Python reference predictions. `scripts/import_evaluation.py` imports the matching local evaluation ZIP; adapt its explicit source path if needed.

`npm test` checks input rejection, stable softmax, post-average bias semantics and float32 RR scaling. Development-only `/qa.html` compares every final probability for 24 inferences (12 beats × 2 schemes) against `artifacts/reference_predictions.json`, with an absolute tolerance of 1e-5. The reference fixture and test harness are excluded from the production Vite entry points. The preparation script confirms that all four ONNX models run with batch size 1.

Verified in this delivery: 24 browser inferences passed with maximum absolute probability difference **5.619055565286857e-8**. Both model schemes, tensor upload, invalid-input rejection, beat reset behavior and PDF download were exercised in the production build under `/kardia/`. Responsive layouts were inspected at 390 px and 1440 px without horizontal overflow. See `artifacts/verification.json`.

Dependency audit: the production dependency audit is clean. The full audit currently reports four development-tool advisories originating in `picomatch` (including transitive Vite tooling). The registry's available latest version, 4.0.7, is still flagged; `npm audit fix` does not resolve them. Do not expose the Vite development server publicly; deployment serves only static `dist/` files. Recheck `npm audit` when upgrading dependencies. jsPDF is pinned through the lockfile to patched 4.2.1.

## Scientific disclaimer

KARDIA is a research prototype for ECG analysis and is not a certified medical diagnostic device.

Retrospective performance, class imbalance, weak S/F recognition, expert R-peak dependence and unvalidated out-of-distribution behavior limit interpretation. Do not use this prototype for diagnosis or treatment decisions.
