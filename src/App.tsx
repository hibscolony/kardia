import { useState, useEffect, useRef, type ReactNode } from "react";
import {
  Activity,
  ArrowUpRight,
  ArrowRight,
  LayoutDashboard,
  ScanLine,
  ChartNoAxesCombined,
  Workflow,
  BookOpen,
  ShieldCheck,
  ChevronRight,
  ChevronLeft,
  Upload,
  Download,
  Play,
  Check,
  Info,
  Cpu,
  Layers,
  Database,
  Menu,
  X,
  LoaderCircle,
  HeartPulse,
  SlidersHorizontal,
  ExternalLink,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Legend,
} from "recharts";
import {
  validateDataset,
  loadModels,
  infer,
  labels,
  names,
  type Dataset,
  type Result,
  type Scheme,
} from "./inference";
const base = import.meta.env.BASE_URL;
const disclaimer =
  "KARDIA is a research prototype for ECG analysis and is not a certified medical diagnostic device.";
const nav = [
  { id: "overview", name: "Overview", icon: LayoutDashboard },
  { id: "analyzer", name: "ECG Analyzer", icon: ScanLine },
  { id: "performance", name: "Model Performance", icon: ChartNoAxesCombined },
  { id: "architecture", name: "How KARDIA Works", icon: Workflow },
  { id: "research", name: "Research & Docs", icon: BookOpen },
];
const colors = ["#52e6b5", "#7695ff", "#12d5d1", "#e3b775"];
function Tag({
  children,
  muted = false,
}: {
  children: ReactNode;
  muted?: boolean;
}) {
  return (
    <span className={"tag " + (muted ? "muted" : "")}>
      <span className="dot" />
      {children}
    </span>
  );
}
function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <section className={"panel " + className}>{children}</section>;
}
function Heading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="page-heading">
      <div className="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  );
}
function Signal({
  samples,
  height = 200,
  unit = "mV",
  seconds = true,
}: {
  samples: number[];
  height?: number;
  unit?: string;
  seconds?: boolean;
}) {
  return (
    <div
      className="signal"
      style={{ height }}
      role="img"
      aria-label={"Actual ECG waveform, " + samples.length + " samples"}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={samples.map((v, i) => ({ t: seconds ? i / 360 : i, v }))}
          margin={{ top: 15, right: 12, bottom: 0, left: -20 }}
        >
          <CartesianGrid stroke="#244255" strokeOpacity={0.48} />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickCount={9}
            tickFormatter={(v) => (seconds ? `${v.toFixed(1)}s` : String(v))}
            stroke="#7f94a9"
            tickLine={false}
            axisLine={false}
            fontSize={10}
          />
          <YAxis
            width={48}
            stroke="#7f94a9"
            tickLine={false}
            axisLine={false}
            fontSize={10}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <Tooltip
            contentStyle={{
              background: "#10243b",
              border: "1px solid #2a435d",
              borderRadius: 12,
              color: "#fff",
            }}
            labelFormatter={(v) =>
              seconds ? `${Number(v).toFixed(3)} s` : `Sample ${v}`
            }
            formatter={(v) => [Number(v).toFixed(4) + " " + unit, "Amplitude"]}
          />
          <Line
            type="linear"
            dataKey="v"
            stroke="#30ddd4"
            strokeWidth={1.7}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
function ModelCards({ go }: { go: () => void }) {
  return (
    <div className="model-grid">
      {([4, 3] as const).map((n, i) => (
        <Panel key={n} className={"model-card model-" + n}>
          <div className="row">
            <span className="square-icon">
              <Layers size={20} />
            </span>
            <span className="model-code">
              MODEL {i ? "B" : "A"} <span>/ {n} CLASSES</span>
            </span>
            <ArrowUpRight size={18} />
          </div>
          <h3>
            {n === 4
              ? "A broader cardiac perspective."
              : "Focused. Refined. Precise."}
          </h3>
          <p>
            {n === 4
              ? "Normal, supraventricular, ventricular & fusion"
              : "Normal, supraventricular & ventricular"}
          </p>
          <div className="metrics">
            <div>
              <strong>
                {n === 4 ? "93.16" : "95.26"}
                <small>%</small>
              </strong>
              <span>Test accuracy</span>
            </div>
            <div>
              <strong>
                {n === 4 ? "63.10" : "74.77"}
                <small>%</small>
              </strong>
              <span>Macro F1 score</span>
            </div>
          </div>
          <div className="model-bottom">
            <div className="label-chips">
              {labels.slice(0, n).map((l, j) => (
                <span key={l} style={{ color: colors[j] }}>
                  {l}
                </span>
              ))}
            </div>
            <button className="text-button" onClick={go}>
              Explore results <ArrowRight size={14} />
            </button>
          </div>
        </Panel>
      ))}
    </div>
  );
}
function Overview({
  go,
  wave,
  status,
}: {
  go: (s: string) => void;
  wave: number[];
  status: string;
}) {
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">
            <span className="tiny-line" /> PRECISION THROUGH INTELLIGENCE
          </div>
          <h1>
            Intelligence behind
            <br />
            every <span>heartbeat.</span>
          </h1>
          <p>
            A clearer view of cardiac rhythms. Explore real ECG signals with
            multi-scale deep learning and transparent, research-backed insights.
          </p>
          <div className="hero-actions">
            <button className="primary" onClick={() => go("analyzer")}>
              Open ECG Analyzer <ArrowUpRight size={18} />
            </button>
            <button className="secondary" onClick={() => go("architecture")}>
              <Play size={14} /> See how it works
            </button>
          </div>
          <div className="hero-foot">
            <ShieldCheck size={14} /> On-device inference <i /> Your data stays
            in your browser
          </div>
        </div>
        <div className="hero-visual">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="visual-top">
            <span className="dot" /> CARDIAC INTELLIGENCE <span>01 / 04</span>
          </div>
          <div className="heart-core">
            <HeartPulse strokeWidth={0.8} />
          </div>
          <div className="visual-wave">
            {wave.length > 0 && (
              <svg
                viewBox="0 0 460 110"
                preserveAspectRatio="none"
                aria-label="MIT-BIH signal excerpt"
              >
                <polyline
                  fill="none"
                  stroke="#55eadc"
                  strokeWidth="1.5"
                  points={wave
                    .slice(400, 1120)
                    .map((v, i) => `${(i / 720) * 460},${78 - v * 54}`)
                    .join(" ")}
                />
              </svg>
            )}
          </div>
          <div className="floating-card float-left">
            <span className="square-icon">
              <Activity size={19} />
            </span>
            <div>
              <small>SIGNAL RESOLUTION</small>
              <strong>
                360 <em>Hz</em>
              </strong>
            </div>
            <span className="mini-bars">▂▅▃▆▄▇</span>
          </div>
          <div className="floating-card float-right">
            <div>
              <span className="dot" /> BROWSER-NATIVE AI
            </div>
            <strong>Two seeds. One ensemble.</strong>
            <small>Multi-scale CNN + attention</small>
          </div>
          <div className="visual-bottom">
            REAL SIGNALS. MEANINGFUL INSIGHTS.<span>+ + +</span>
          </div>
        </div>
      </section>
      <div className="section-heading">
        <div>
          <span className="eyebrow">THE RESEARCH AT A GLANCE</span>
          <h2>Two models. One thoughtful approach.</h2>
        </div>
        <button className="text-button" onClick={() => go("performance")}>
          View full evaluation <ArrowUpRight size={16} />
        </button>
      </div>
      <ModelCards go={() => go("performance")} />
      <div className="inline-note">
        <Info size={15} />
        <span>
          Different class scopes, different experiments. Fusion (F) beats are
          excluded from the three-class model.
        </span>
      </div>
      <div className="overview-bottom">
        <Panel className="wave-panel">
          <div className="row">
            <div>
              <div className="eyebrow">A WINDOW INTO THE SIGNAL</div>
              <h3>Every beat tells a story.</h3>
            </div>
            <Tag>MIT-BIH · Record 100</Tag>
          </div>
          {wave.length ? (
            <Signal samples={wave.slice(0, 1440)} height={164} />
          ) : (
            <div className="empty">Loading source waveform…</div>
          )}
          <div className="wave-caption">
            <span>Lead MLII · 360 Hz · amplitude in mV</span>
            <button className="text-button" onClick={() => go("analyzer")}>
              Explore signal <ArrowRight size={14} />
            </button>
          </div>
        </Panel>
        <Panel className="readiness">
          <div className="square-icon">
            <Cpu size={21} />
          </div>
          <h3>Intelligence, locally.</h3>
          <p>
            Original ONNX models. Private, in-browser analysis. No server
            required.
          </p>
          <div className="readiness-line">
            <span>Model artifacts</span>
            <span className="good">4 supplied</span>
          </div>
          <div className="readiness-line">
            <span>Inference engine</span>
            <span>{status}</span>
          </div>
          <button className="text-button" onClick={() => go("analyzer")}>
            Launch workspace <ArrowRight size={15} />
          </button>
        </Panel>
      </div>
    </>
  );
}
function Performance() {
  const data = labels.map((name, i) => ({
    name,
    "Four classes": [96.44, 28.53, 91.26, 36.19][i],
    "Three classes": [97.48, 33.24, 93.6, null][i],
  }));
  return (
    <>
      <Heading
        eyebrow="EVIDENCE, NOT ASSUMPTIONS"
        title="Behind the performance."
        description="Two held-out experiments. Transparent metrics. A clear view of what the models can—and cannot—recognize."
      />
      <div className="metric-strip">
        {[
          ["47,549", "Four-class test beats"],
          ["47,162", "Three-class test beats"],
          ["21", "Held-out records"],
          ["2", "Seeds per ensemble"],
        ].map(([v, l]) => (
          <Panel key={l}>
            <strong>{v}</strong>
            <span>{l}</span>
          </Panel>
        ))}
      </div>
      <div className="two-col">
        <Panel>
          <div className="eyebrow">EXPERIMENT COMPARISON</div>
          <h3>Look beyond accuracy.</h3>
          <ResponsiveContainer width="100%" height={270}>
            <BarChart
              data={[
                {
                  name: "Accuracy",
                  "Four classes": 93.16,
                  "Three classes": 95.26,
                },
                {
                  name: "Macro F1",
                  "Four classes": 63.1,
                  "Three classes": 74.77,
                },
                {
                  name: "Weighted F1",
                  "Four classes": 93.06,
                  "Three classes": 94.79,
                },
              ]}
            >
              <CartesianGrid vertical={false} stroke="#20354c" />
              <XAxis dataKey="name" stroke="#8fa5bb" fontSize={11} />
              <YAxis domain={[0, 100]} stroke="#8fa5bb" fontSize={11} />
              <Tooltip
                contentStyle={{ background: "#10243b", borderColor: "#2a435d" }}
              />
              <Legend />
              <Bar
                dataKey="Four classes"
                fill="#397bff"
                radius={[5, 5, 0, 0]}
              />
              <Bar
                dataKey="Three classes"
                fill="#12d5d1"
                radius={[5, 5, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel>
          <div className="eyebrow">CLASS-LEVEL F1 · %</div>
          <h3>Not all rhythms are equally easy.</h3>
          <ResponsiveContainer width="100%" height={270}>
            <BarChart data={data}>
              <CartesianGrid vertical={false} stroke="#20354c" />
              <XAxis dataKey="name" stroke="#8fa5bb" />
              <YAxis domain={[0, 100]} stroke="#8fa5bb" fontSize={11} />
              <Tooltip
                contentStyle={{ background: "#10243b", borderColor: "#2a435d" }}
              />
              <Legend />
              <Bar
                dataKey="Four classes"
                fill="#397bff"
                radius={[5, 5, 0, 0]}
              />
              <Bar
                dataKey="Three classes"
                fill="#12d5d1"
                radius={[5, 5, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>
      <div className="two-col">
        <Panel>
          <div className="eyebrow">TEST SET COMPOSITION</div>
          <h3>A highly imbalanced dataset.</h3>
          {labels.map((l, i) => (
            <div className="distribution" key={l}>
              <span style={{ color: colors[i] }}>{l}</span>
              <div>
                <div
                  style={{
                    width: `${([42179, 1782, 3201, 387][i] / 47549) * 100}%`,
                    background: colors[i],
                  }}
                />
              </div>
              <strong>{[42179, 1782, 3201, 387][i].toLocaleString()}</strong>
            </div>
          ))}
          <p className="small">
            The three-class experiment uses the same N/S/V counts and excludes
            all 387 fusion beats.
          </p>
        </Panel>
        <Panel>
          <div className="eyebrow">INTERPRETING THE RESULTS</div>
          <h3>Confidence needs context.</h3>
          <p>
            High aggregate accuracy is influenced by the dominant N class.
            Supraventricular F1 remains low in both experiments. Fusion
            recognition is limited in the four-class model.
          </p>
          <p>
            The four-class macro F1 target of 0.70 was not met. The three-class
            experiment meets that target on a narrower label scope.
          </p>
          <div className="notice">
            <Info size={17} />
            <span>
              These are retrospective research results, not prospective clinical
              validation.
            </span>
          </div>
        </Panel>
      </div>
      <Panel>
        <div className="eyebrow">CONFUSION MATRICES</div>
        <h3>Original notebook evaluation figures.</h3>
        <ConfusionFigures />
      </Panel>
    </>
  );
}
function ConfusionFigures() {
  const [figs, setFigs] = useState<string[]>([]);
  useEffect(() => {
    fetch(base + "data/figures.json")
      .then((r) => r.json())
      .then(setFigs)
      .catch(() => {});
  }, []);
  return figs.length ? (
    <div className="figure-grid">
      {figs.map((f, i) => (
        <figure key={f}>
          <img
            src={base + "data/" + f}
            alt={`Original notebook evaluation figure ${i + 1}`}
          />
          <figcaption>
            Unmodified output from the supplied research notebook
          </figcaption>
        </figure>
      ))}
    </div>
  ) : (
    <p>
      Exact confusion-matrix cells were not supplied as structured data. They
      are not reconstructed from rounded F1 scores.
    </p>
  );
}
const stages = [
  [
    "ECG signal",
    "Two leads. One cardiac story.",
    "MIT-BIH recordings sampled at 360 Hz. Expert beat annotations provide R-peak positions; this is not an automatic R-peak detector.",
  ],
  [
    "Signal preprocessing",
    "Preserve morphology. Normalize consistently.",
    "Third-order Butterworth bandpass, 0.5–40 Hz, zero-phase sosfiltfilt. Extract 112 samples before and 144 after each annotated peak; subtract each lead’s median, scale by its 95th absolute percentile (minimum 0.05), clip to ±5.",
  ],
  [
    "Multi-scale CNN",
    "Read patterns at multiple scales.",
    "Parallel convolution branches extract complementary morphological patterns from current and previous two-lead beats, arranged as a 4 × 256 tensor.",
  ],
  [
    "Temporal self-attention",
    "Connect the moments within a beat.",
    "Temporal attention integrates relationships across the learned ECG sequence rather than relying on a single local feature.",
  ],
  [
    "RR cross-attention fusion",
    "Bring rhythm into the picture.",
    "Eight causal RR features summarize up to eight preceding intervals. The training scaler standardizes each feature and values are clipped to ±8; RR tokens interact with ECG features.",
  ],
  [
    "Classification",
    "Two seeds. Bias-aware ensemble.",
    "Softmax each seed’s logits, average the probabilities, then apply softmax(log(max(mean, 1e-8)) + the validation-selected logit bias).",
  ],
  [
    "Results",
    "Make uncertainty visible.",
    "Display final class scores, normalized entropy and seed disagreement. Model scores are not calibrated clinical probabilities.",
  ],
];
function Architecture() {
  const [stage, setStage] = useState(0);
  return (
    <>
      <Heading
        eyebrow="FROM SIGNAL TO INSIGHT"
        title="Anatomy of the intelligence."
        description="Follow a heartbeat through the original multi-scale CNN and attention-based research architecture."
      />
      <Panel className="architecture">
        <div className="pipeline">
          {stages.map((s, i) => (
            <button
              className={stage === i ? "selected" : ""}
              key={s[0]}
              onClick={() => setStage(i)}
            >
              <span>{String(i + 1).padStart(2, "0")}</span>
              {s[0]}
              <ChevronRight size={16} />
            </button>
          ))}
        </div>
        <AnimatePresence mode="wait">
          <motion.div
            className="stage-detail"
            key={stage}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="stage-number">0{stage + 1}</div>
            <span className="eyebrow">{stages[stage][0]}</span>
            <h2>{stages[stage][1]}</h2>
            <p>{stages[stage][2]}</p>
            <div className="tensor-pills">
              <span>ECG [1, 4, 256]</span>
              <span>RR [1, 8]</span>
              <span>WASM execution</span>
            </div>
            <button
              className="secondary"
              onClick={() => setStage((stage + 1) % 7)}
            >
              Next component <ArrowRight size={16} />
            </button>
          </motion.div>
        </AnimatePresence>
      </Panel>
      <div className="inline-note">
        <ShieldCheck size={16} />
        The browser runs the supplied ONNX checkpoints. No replacement model or
        synthetic inference.
      </div>
    </>
  );
}
function Research() {
  return (
    <>
      <Heading
        eyebrow="OPEN ABOUT THE SCIENCE"
        title="Research with a clear foundation."
        description="Methods, provenance and limitations from the supplied KARDIA V3 research notebook."
      />
      <div className="two-col">
        <Panel>
          <BookOpen className="accent" />
          <h3>Research objective</h3>
          <p>
            Investigate heartbeat classification using multi-scale morphology,
            temporal self-attention and RR cross-attention fusion. Compare
            four-class N/S/V/F and three-class N/S/V experiments without using
            the test set for tuning.
          </p>
          <h3>Data & provenance</h3>
          <p>
            MIT-BIH Arrhythmia Database. Demo record 100 is processed with the
            exact extraction function from the supplied notebook. Signals are
            not generated.
          </p>
          <a
            className="text-button"
            href="https://physionet.org/content/mitdb/1.0.0/"
            target="_blank"
            rel="noreferrer"
          >
            PhysioNet · MIT-BIH v1.0.0 <ExternalLink size={14} />
          </a>
          <p className="small">
            Moody, G. & Mark, R. (2005). MIT-BIH Arrhythmia Database, version
            1.0.0. DOI: 10.13026/C2F305. Data attribution: Open Data Commons
            Attribution License v1.0.
          </p>
        </Panel>
        <Panel>
          <Database className="accent" />
          <h3>Record-disjoint evaluation</h3>
          <div className="protocol">
            <span>TRAINING</span>
            <p>16 DS1 records; record 208 retained.</p>
            <span>VALIDATION</span>
            <p>223, 207, 205 · selection and logit bias tuning.</p>
            <span>CALIBRATION</span>
            <p>124, 118, 106 · held out from training.</p>
            <span>TESTING</span>
            <p>
              21 DS2 records. Record 202 excluded to avoid same-patient overlap
              with record 201.
            </p>
          </div>
        </Panel>
      </div>
      <Panel>
        <div className="eyebrow">REPRODUCIBILITY</div>
        <h3>Exactly what goes into the model.</h3>
        <div className="two-col inner">
          <div>
            <p>
              Channel order: current lead 1, current lead 2, previous lead 1,
              previous lead 2. Lead 1 uses MLII when available. Each beat
              contains 256 samples per channel.
            </p>
            <p>
              RR order: previous interval, second previous interval, their
              ratio, previous/median ratio, median, mean, coefficient of
              variation, and previous minus second previous. Statistics use up
              to eight causal intervals.
            </p>
          </div>
          <div>
            <p>
              Two seeds per scheme: 42 and 123. Original training scalers and
              metadata are bundled. Model loading and inference use WebAssembly
              in the browser.
            </p>
            <p>
              Notebook export logs report PyTorch–ONNX maximum absolute logit
              error below 10⁻⁶ on the export test. Browser parity is tested
              separately with real demo beats.
            </p>
            <a
              className="text-button"
              href={base + "models/model_metadata.json"}
              download
            >
              Download model metadata <Download size={14} />
            </a>
          </div>
        </div>
      </Panel>
      <div className="two-col">
        <Panel>
          <h3>Known limitations</h3>
          <ul className="research-list">
            <li>
              Expert R-peak annotations were used; arbitrary raw ECG upload is
              not validated.
            </li>
            <li>
              Class imbalance and weak S/F recognition limit generalization.
            </li>
            <li>Three-class models cannot recognize F; F was excluded.</li>
            <li>
              Conformal test coverage in the notebook was below the nominal 90%;
              no clinical coverage guarantee is made.
            </li>
            <li>
              Retrospective MIT-BIH evaluation does not establish medical-device
              safety or performance in new populations.
            </li>
          </ul>
        </Panel>
        <Panel>
          <h3>Research team & institution</h3>
          <p>
            Names and institutional affiliations were not provided in the
            supplied research artifacts.
          </p>
          <h3>Supplied artifacts</h3>
          <p>
            notebook9963638c15.ipynb
            <br />
            Four ONNX model files
            <br />
            Two RR training scalers
            <br />
            model_metadata.json
          </p>
          <div className="notice">
            <ShieldCheck size={18} />
            <span>{disclaimer}</span>
          </div>
        </Panel>
      </div>
    </>
  );
}
function Analyzer({
  notify,
  setGlobalStatus,
}: {
  notify: (s: string) => void;
  setGlobalStatus: (s: string) => void;
}) {
  const [scheme, setScheme] = useState<Scheme>(4),
    [data, setData] = useState<Dataset | null>(null),
    [beat, setBeat] = useState(0),
    [lead, setLead] = useState(0),
    [result, setResult] = useState<Result | null>(null),
    [status, setStatus] = useState("Not loaded"),
    [loading, setLoading] = useState(false),
    [running, setRunning] = useState(false),
    [ready, setReady] = useState(false),
    [error, setError] = useState("");
  const file = useRef<HTMLInputElement>(null);
  const generation = useRef(0);
  useEffect(() => {
    let live = true;
    setReady(false);
    setResult(null);
    setLoading(true);
    setError("");
    loadModels(scheme, (s) => {
      if (live) {
        setStatus(s);
        setGlobalStatus(s);
      }
    })
      .then(() => {
        if (live) setReady(true);
      })
      .catch((e) => {
        if (live) {
          setStatus("AI Model Not Connected");
          setGlobalStatus("Not connected");
          setError(String(e));
        }
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
      generation.current++;
    };
  }, [scheme]);
  function replace(d: Dataset) {
    generation.current++;
    setData(d);
    setBeat(0);
    setResult(null);
    setError("");
  }
  async function demo() {
    try {
      const r = await fetch(base + "data/demo.json");
      if (!r.ok) throw Error("Demo data could not be loaded");
      replace(validateDataset(await r.json()));
      notify("Real MIT-BIH demo loaded · notebook preprocessing");
    } catch (e) {
      setError(String(e));
      notify(String(e));
    }
  }
  async function upload(f: File | undefined) {
    if (!f) return;
    try {
      if (f.size > 30 * 1024 * 1024) throw Error("Maximum file size is 30 MB.");
      replace(validateDataset(JSON.parse(await f.text())));
      notify("Tensor file validated and loaded");
    } catch (e) {
      setError(String(e));
      notify(String(e));
    } finally {
      if (file.current) file.current.value = "";
    }
  }
  async function run() {
    if (!data || !ready) return;
    const id = ++generation.current;
    setRunning(true);
    setResult(null);
    setError("");
    try {
      const r = await infer(data, beat, scheme);
      if (id === generation.current) {
        setResult(r);
        notify("Ensemble analysis complete");
      }
    } catch (e) {
      if (id === generation.current) {
        setError(String(e));
        notify(String(e));
      }
    } finally {
      setRunning(false);
    }
  }
  function changeBeat(n: number) {
    generation.current++;
    setBeat(n);
    setResult(null);
    setError("");
  }
  async function report() {
    if (!result || !data) return;
    try {
      const { jsPDF } = await import("jspdf");
      const pdf = new jsPDF();
      pdf.setFontSize(25);
      pdf.text("KARDIA", 20, 25);
      pdf.setFontSize(13);
      pdf.text("ECG research analysis report", 20, 35);
      pdf.setFontSize(10);
      const lines = [
        `Generated: ${result.timestamp}`,
        `Source: ${result.source}`,
        `Beat index: ${result.beat + 1} | Scheme: ${result.scheme} classes | Seeds: 42, 123`,
        `Prediction: ${labels[result.prediction]} - ${names[result.prediction]}`,
        `Reference annotation: ${data.annotations?.[beat] ?? "Not supplied"}`,
        ...result.probabilities.map(
          (p, i) => `${labels[i]}: ${(p * 100).toFixed(4)}%`,
        ),
        `Normalized entropy: ${result.entropy.toFixed(4)}`,
        `Seed disagreement (total variation): ${result.disagreement.toFixed(4)}`,
        `Inference: ${result.elapsed.toFixed(0)} ms | ONNX Runtime WASM`,
        "Preprocessing: KARDIA V3 notebook; scheme-specific RR training scaler.",
        "Ensemble: mean seed softmax probabilities, then metadata logit bias.",
        "Model scores are not calibrated clinical probabilities.",
        disclaimer,
      ];
      let y = 48;
      for (const line of lines) {
        const wrap = pdf.splitTextToSize(line, 170);
        pdf.text(wrap, 20, y);
        y += wrap.length * 5 + 5;
      }
      pdf.save(`KARDIA-${result.scheme}class-beat${beat + 1}.pdf`);
      notify("PDF report downloaded");
    } catch (e) {
      notify("Report export failed: " + String(e));
    }
  }
  return (
    <>
      <Heading
        eyebrow="YOUR CARDIAC ANALYSIS WORKSPACE"
        title="A closer look at every beat."
        description="Explore annotated MIT-BIH data or upload notebook-preprocessed tensors. All analysis stays on this device."
      />
      <div className="workspace-toolbar">
        <div className="segmented">
          {([4, 3] as const).map((n) => (
            <button
              disabled={running || loading}
              onClick={() => {
                generation.current++;
                setScheme(n);
              }}
              className={scheme === n ? "active" : ""}
              key={n}
            >
              {n}-class model{" "}
              <span>{n === 4 ? "N · S · V · F" : "N · S · V"}</span>
            </button>
          ))}
        </div>
        <Tag muted={!ready}>{status}</Tag>
      </div>
      {loading && (
        <div className="loading-track">
          <span />
        </div>
      )}
      {error && (
        <div role="alert" className="notice error">
          <Info size={18} />
          <span>{error}</span>
          {!ready && !loading && (
            <button
              className="text-button"
              onClick={() => {
                setLoading(true);
                loadModels(scheme, setStatus)
                  .then(() => {
                    setReady(true);
                    setError("");
                    setGlobalStatus("Ready · WASM");
                  })
                  .catch((e) => setError(String(e)))
                  .finally(() => setLoading(false));
              }}
            >
              Retry
            </button>
          )}
        </div>
      )}
      <div className="analyzer-grid">
        <div>
          <Panel>
            <div className="row">
              <div>
                <div className="eyebrow">SIGNAL INPUT</div>
                <h3>
                  {data ? "ECG signal explorer" : "Start with a real signal."}
                </h3>
              </div>
              <Activity className="accent" size={22} />
            </div>
            <div className="input-actions">
              <button className="secondary" onClick={demo} disabled={running}>
                <Play size={15} /> Load MIT-BIH demo
              </button>
              <button
                className="secondary"
                onClick={() => file.current?.click()}
                disabled={running}
              >
                <Upload size={15} /> Upload tensor JSON
              </button>
              <input
                ref={file}
                type="file"
                accept=".json,application/json"
                hidden
                onChange={(e) => upload(e.target.files?.[0])}
              />
            </div>
            {data ? (
              <>
                <div className="signal-heading">
                  <span>{data.source}</span>
                  <select
                    aria-label="ECG channel"
                    value={lead}
                    onChange={(e) => setLead(Number(e.target.value))}
                  >
                    <option value={0}>Current · lead 1</option>
                    <option value={1}>Current · lead 2</option>
                    <option value={2}>Previous · lead 1</option>
                    <option value={3}>Previous · lead 2</option>
                  </select>
                </div>
                <Signal
                  samples={data.ecg[beat][lead]}
                  height={260}
                  seconds={false}
                  unit="normalized"
                />
                <div className="beat-nav">
                  <button
                    className="icon-button"
                    aria-label="Previous beat"
                    disabled={beat === 0 || running}
                    onClick={() => changeBeat(beat - 1)}
                  >
                    <ChevronLeft size={18} />
                  </button>
                  <div>
                    Beat <strong>{beat + 1}</strong> / {data.ecg.length}
                    <small>
                      Reference annotation:{" "}
                      {data.annotations?.[beat] ?? "Not supplied"}
                      {data.beatIds
                        ? ` · record beat index ${data.beatIds[beat]}`
                        : ""}
                    </small>
                  </div>
                  <button
                    className="icon-button"
                    aria-label="Next beat"
                    disabled={beat === data.ecg.length - 1 || running}
                    onClick={() => changeBeat(beat + 1)}
                  >
                    <ChevronRight size={18} />
                  </button>
                </div>
                <input
                  className="beat-slider"
                  aria-label="Choose beat"
                  type="range"
                  min={0}
                  max={data.ecg.length - 1}
                  value={beat}
                  disabled={running}
                  onChange={(e) => changeBeat(Number(e.target.value))}
                />
                <div className="signal-stats">
                  <div>
                    <span>SAMPLE RATE</span>
                    <strong>360 Hz</strong>
                  </div>
                  <div>
                    <span>ECG TENSOR</span>
                    <strong>4 × 256</strong>
                  </div>
                  <div>
                    <span>RR FEATURES</span>
                    <strong>8 temporal</strong>
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <div className="empty-orbit">
                  <ScanLine size={36} />
                </div>
                <h3>Your next insight starts here.</h3>
                <p>
                  Load the real MIT-BIH demo to inspect each beat and run the
                  original research models.
                </p>
                <span>
                  Two leads · Expert annotations · Original preprocessing
                </span>
              </div>
            )}
            <div className="notice">
              <ShieldCheck size={17} />
              <span>
                Only notebook-preprocessed ECG tensors are accepted. Raw ECG,
                CSV, PDFs and ECG images are not supported.
              </span>
            </div>
          </Panel>
          <details className="panel input-spec">
            <summary>
              <SlidersHorizontal size={17} /> Input format & preprocessing
            </summary>
            <p>
              Upload a JSON object containing version: “kardia-tensors-v1”,
              preprocessing: “kardia-v3-notebook”, rrFormat: “raw”,
              samplingRate: 360, source, ecg [beats,4,256] and rr [beats,8]. ECG
              is already filtered and normalized; the browser applies the
              original RR scaler for the selected scheme.
            </p>
            <p>
              Shape checks cannot establish clinical signal validity. Use the
              supplied preparation script with the original notebook.
            </p>
            <a
              className="text-button"
              href={base + "data/demo.json"}
              download="kardia-demo-tensors.json"
            >
              Download valid demo JSON <Download size={15} />
            </a>
          </details>
        </div>
        <Panel className="prediction">
          <div className="row">
            <div className="eyebrow">AI PREDICTION</div>
            <Cpu size={20} />
          </div>
          {result ? (
            <>
              <div
                className="result-class"
                style={{ color: colors[result.prediction] }}
              >
                {labels[result.prediction]}
              </div>
              <h3>{names[result.prediction]}</h3>
              <div className="confidence">
                {(result.probabilities[result.prediction] * 100).toFixed(2)}
                <span>%</span>
              </div>
              <p className="small">
                Final ensemble score · not clinical certainty
              </p>
              <div className="probabilities">
                {result.probabilities.map((p, i) => (
                  <div key={i}>
                    <div className="row">
                      <span>
                        {labels[i]} <small>{names[i]}</small>
                      </span>
                      <strong>{(p * 100).toFixed(2)}%</strong>
                    </div>
                    <div className="prob-track">
                      <div
                        style={{ width: p * 100 + "%", background: colors[i] }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <div className="uncertainty">
                <div>
                  <span>Normalized entropy</span>
                  <strong>{result.entropy.toFixed(3)}</strong>
                </div>
                <div>
                  <span>Seed disagreement</span>
                  <strong>{(result.disagreement * 100).toFixed(2)}%</strong>
                </div>
                <div>
                  <span>Inference time</span>
                  <strong>{result.elapsed.toFixed(0)} ms</strong>
                </div>
              </div>
              <p className="small">
                Entropy measures score spread (0–1). Disagreement is total
                variation between seed probabilities before bias. Neither is a
                validated clinical threshold.
              </p>
            </>
          ) : (
            <div className="prediction-empty">
              <div className="prediction-ring">
                <Cpu size={28} />
              </div>
              <h3>Ready for discovery.</h3>
              <p>
                {data
                  ? "Your signal is loaded. Run the ensemble to reveal the model’s interpretation."
                  : "Load a signal to see class probabilities, model scores and uncertainty."}
              </p>
              <div className="ghost-probs">
                {labels.slice(0, scheme).map((l) => (
                  <div key={l}>
                    <span>{l}</span>
                    <i />
                    <span>—</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <button
            className="primary wide"
            onClick={run}
            disabled={
              !data ||
              !ready ||
              running ||
              loading ||
              (scheme === 3 && data?.annotations?.[beat] === "F")
            }
          >
            {running ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <ScanLine size={17} />
            )}{" "}
            {running ? "Analyzing both seeds…" : "Analyze heartbeat"}
            <ArrowRight size={16} />
          </button>
          {scheme === 3 && data?.annotations?.[beat] === "F" && (
            <p className="small">
              Fusion is outside this model’s scope. Select four classes.
            </p>
          )}
          <button
            className="secondary wide"
            disabled={!result || running}
            onClick={report}
          >
            <Download size={16} /> Download PDF report
          </button>
          <div className="local-note">
            <ShieldCheck size={13} /> Private by design. Processed locally.
          </div>
        </Panel>
      </div>
    </>
  );
}
export default function App() {
  const getPage = () =>
    nav.some((n) => n.id === location.hash.slice(1))
      ? location.hash.slice(1)
      : "overview";
  const [page, setPage] = useState(getPage),
    [mobile, setMobile] = useState(false),
    [wave, setWave] = useState<number[]>([]),
    [toast, setToast] = useState(""),
    [modelStatus, setModelStatus] = useState("Not loaded");
  useEffect(() => {
    const f = () => setPage(getPage());
    window.addEventListener("hashchange", f);
    fetch(base + "data/waveform.json")
      .then((r) => r.json())
      .then((d) => setWave(d.samples))
      .catch(() => setToast("Waveform could not be loaded. Please refresh."));
    return () => window.removeEventListener("hashchange", f);
  }, []);
  useEffect(() => {
    if (toast) {
      const t = setTimeout(() => setToast(""), 5000);
      return () => clearTimeout(t);
    }
  }, [toast]);
  function go(s: string) {
    location.hash = s;
    setPage(s);
    setMobile(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      {mobile && (
        <button
          aria-label="Close navigation"
          className="sidebar-overlay"
          onClick={() => setMobile(false)}
        />
      )}
      <aside className={"sidebar " + (mobile ? "open" : "")}>
        <a className="brand" href="#overview" onClick={() => go("overview")}>
          <span className="brand-icon">
            <Activity size={25} />
          </span>
          <span>
            KARDIA<small>INTELLIGENT ECG ANALYSIS</small>
          </span>
        </a>
        <div className="workspace-label">
          RESEARCH WORKSPACE <span>V1.0</span>
        </div>
        <nav>
          {nav.map(({ id, name, icon: Icon }) => (
            <button
              key={id}
              onClick={() => go(id)}
              aria-current={page === id ? "page" : undefined}
              className={page === id ? "active" : ""}
            >
              <Icon size={19} />
              <span>{name}</span>
              {page === id && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-science">
          <div className="science-icon">
            <HeartPulse size={24} />
          </div>
          <span>
            Built on science.
            <br />
            Designed for insight.
          </span>
          <p>
            From cardiac signals to
            <br />
            meaningful understanding.
          </p>
          <button className="text-button" onClick={() => go("research")}>
            Explore the research <ArrowUpRight size={14} />
          </button>
        </div>
        <div className="sidebar-bottom">
          <div className="status-orb" />
          <div>
            Browser-native AI<small>Private. Local. Transparent.</small>
          </div>
          <ShieldCheck size={17} />
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              onClick={() => setMobile(true)}
            >
              <Menu size={21} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>{nav.find((n) => n.id === page)?.name}</strong>
          </div>
          <div className="top-right">
            <span className="research-badge">RESEARCH PROTOTYPE</span>
            <span className="header-separator" />
            <span className="avatar">K</span>
          </div>
        </header>
        <main id="main">
          <div className="welcome-row">
            <span>
              <span className="dot" /> CARDIAC INTELLIGENCE WORKSPACE
            </span>
            <span>MIT-BIH RESEARCH · V3</span>
          </div>
          <AnimatePresence mode="wait">
            <motion.div
              key={page}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              {page === "overview" ? (
                <Overview go={go} wave={wave} status={modelStatus} />
              ) : page === "analyzer" ? (
                <Analyzer notify={setToast} setGlobalStatus={setModelStatus} />
              ) : page === "performance" ? (
                <Performance />
              ) : page === "architecture" ? (
                <Architecture />
              ) : (
                <Research />
              )}
            </motion.div>
          </AnimatePresence>
          <footer>
            <ShieldCheck size={15} />
            <span>{disclaimer}</span>
            <span className="footer-brand">KARDIA © 2026</span>
          </footer>
        </main>
      </div>
      <AnimatePresence>
        {toast && (
          <motion.div
            role="status"
            className="toast"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <Info size={17} />
            <span>{toast}</span>
            <button
              aria-label="Dismiss notification"
              onClick={() => setToast("")}
            >
              <X size={16} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
