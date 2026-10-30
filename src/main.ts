// Browser entry point. Everything runs locally: the model and metrics are fetched
// from this site once, and messages are scored in this tab without any network call.
import { loadModel, type Model } from "./model";
import { describeTerm, explain, formatConfidence, type Explanation } from "./signals";

interface Metrics {
  dataset: { name: string; source: string; messages: number; ham: number; spam: number };
  split: { train: number; test: number; testHam: number; testSpam: number; seed: number };
  model: { vocabularySize: number; C: number; threshold: number; sklearnVersion: string };
  test: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    confusionMatrix: {
      trueNegatives: number;
      falsePositives: number;
      falseNegatives: number;
      truePositives: number;
    };
  };
}

const EXAMPLES: Record<string, { text: string; sender: string }> = {
  smishing: {
    text: "USPS: Your package is on hold due to an unpaid shipping fee. Act now, limited time: https://bit.ly/3Pq9xZ",
    sender: "usps-alert@notify-mail.com",
  },
  prize: {
    text: "URGENT! We are trying to contact U. Todays draw shows that you have won a £800 prize GUARANTEED. Call 09050003091 from land line. Claim C52. Valid 12hrs only",
    sender: "80086",
  },
  legit: {
    text: "Hey, are we still ok for dinner tonight? I'll bring the wine. Running about 10 mins late",
    sender: "",
  },
};

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text?: string,
  className?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;

function render(e: Explanation): void {
  const result = $("result");
  result.hidden = false;
  result.className = `result ${e.score.isSpam ? "spam" : "ham"}`;

  $("verdict").textContent = e.score.isSpam ? "Likely spam / smishing" : "Likely legitimate";
  $("confidence").textContent = `${formatConfidence(e.score.confidence)} confidence`;
  const meter = $("meter-fill");
  meter.style.width = pct(e.score.probability);
  $("meter").setAttribute("aria-valuenow", String(Math.round(e.score.probability * 100)));
  $("probability").textContent =
    `Model spam probability: ${pct(e.score.probability)} (threshold ${pct(0.5, 0)})`;
  $("summary").textContent = e.summary;

  const signals = $("signals");
  signals.replaceChildren();
  if (e.signals.length === 0) {
    signals.append(el("li", "No pattern-based warning signs found.", "muted"));
  }
  for (const s of e.signals) {
    const li = el("li");
    li.append(el("strong", s.label), document.createTextNode(`: ${s.detail}`));
    signals.append(li);
  }

  const fill = (id: string, terms: Explanation["spamTerms"]) => {
    const list = $(id);
    list.replaceChildren();
    if (terms.length === 0) list.append(el("li", "none", "muted"));
    for (const t of terms) {
      const li = el("li");
      li.append(el("code", describeTerm(t.term)), el("span", ` ${t.contribution > 0 ? "+" : ""}${t.contribution.toFixed(2)}`, "weight"));
      list.append(li);
    }
  };
  fill("spam-terms", e.spamTerms);
  fill("ham-terms", e.hamTerms);
}

function renderMetrics(m: Metrics): void {
  const rows: [string, string][] = [
    ["Accuracy", pct(m.test.accuracy)],
    ["Precision (spam)", pct(m.test.precision)],
    ["Recall (spam)", pct(m.test.recall)],
    ["F1 (spam)", pct(m.test.f1)],
  ];
  const body = $("metrics-body");
  body.replaceChildren();
  for (const [k, v] of rows) {
    const tr = el("tr");
    tr.append(el("th", k), el("td", v));
    body.append(tr);
  }
  const cm = m.test.confusionMatrix;
  $("cm-tn").textContent = String(cm.trueNegatives);
  $("cm-fp").textContent = String(cm.falsePositives);
  $("cm-fn").textContent = String(cm.falseNegatives);
  $("cm-tp").textContent = String(cm.truePositives);
  $("metrics-note").textContent =
    `Measured on a held-out stratified 20% test split (${m.split.test} messages: ` +
    `${m.split.testHam} legitimate, ${m.split.testSpam} spam) of the ${m.dataset.messages.toLocaleString("en")} ` +
    `-message SMS Spam Collection. Trained on the other ${m.split.train} (seed ${m.split.seed}); ` +
    `${m.model.vocabularySize.toLocaleString("en")}-term vocabulary, C=${m.model.C}, scikit-learn ${m.model.sklearnVersion}.`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

async function init(): Promise<void> {
  const status = $("status");
  const button = $("analyze") as HTMLButtonElement;
  const input = $("message") as HTMLTextAreaElement;
  const sender = $("sender") as HTMLInputElement;

  let model: Model;
  try {
    const [modelJson, metrics] = await Promise.all([
      fetchJson<unknown>("model.json"),
      fetchJson<Metrics>("metrics.json"),
    ]);
    model = loadModel(modelJson);
    renderMetrics(metrics);
  } catch (err) {
    status.textContent =
      `Could not load the model (${(err as Error).message}). If you opened index.html directly from disk, ` +
      "serve the docs/ folder over HTTP instead (for example: npm run serve).";
    status.className = "status error";
    return;
  }
  status.textContent = `Model loaded: ${model.raw.vocabulary.length.toLocaleString("en")} features. Ready.`;
  button.disabled = false;

  const run = () => {
    const text = input.value.trim();
    if (!text) {
      status.textContent = "Paste a message first.";
      return;
    }
    status.textContent = "Analysed locally in your browser.";
    render(explain(model, text, sender.value));
  };

  button.addEventListener("click", run);
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) run();
  });
  document.querySelectorAll<HTMLButtonElement>("[data-example]").forEach((b) => {
    b.addEventListener("click", () => {
      const ex = EXAMPLES[b.dataset.example ?? ""];
      if (!ex) return;
      input.value = ex.text;
      sender.value = ex.sender;
      run();
    });
  });
}

void init();
