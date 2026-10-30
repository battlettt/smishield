import { analyze } from "./tokenizer";

/** Shape of docs/model.json as exported by train.py. */
export interface ModelJson {
  format: "smishield-tfidf-logreg";
  version: number;
  labels: Record<"0" | "1", string>;
  positiveLabel: string;
  threshold: number;
  tokenizer: {
    lowercase: boolean;
    ngramRange: [number, number];
    urlToken: string;
    shortUrlToken: string;
    currencyToken: string;
    longNumberToken: string;
    longNumberMinDigits: number;
    shortenerDomains: string[];
  };
  tfidf: { sublinearTf: boolean; smoothIdf: boolean; norm: "l2" };
  vocabulary: string[];
  idf: number[];
  coef: number[];
  intercept: number;
}

export interface Model {
  raw: ModelJson;
  index: Map<string, number>;
}

/** A sparse TF-IDF vector: feature index -> weight (l2-normalised). */
export type SparseVector = Map<number, number>;

export interface FeatureContribution {
  term: string;
  weight: number; // tf-idf value in this message
  coef: number; // model coefficient
  contribution: number; // weight * coef, positive pushes towards spam
}

export interface Score {
  label: "spam" | "ham";
  isSpam: boolean;
  probability: number; // P(spam)
  confidence: number; // probability of the predicted label
  logit: number;
  contributions: FeatureContribution[]; // sorted by |contribution| desc
}

export function loadModel(json: unknown): Model {
  const m = json as ModelJson;
  if (!m || m.format !== "smishield-tfidf-logreg") throw new Error("Not a smishield model file");
  const n = m.vocabulary.length;
  if (m.idf.length !== n || m.coef.length !== n) {
    throw new Error(`Model arrays disagree: vocab=${n} idf=${m.idf.length} coef=${m.coef.length}`);
  }
  if (m.tfidf.norm !== "l2") throw new Error(`Unsupported norm ${m.tfidf.norm}`);
  const index = new Map<string, number>();
  m.vocabulary.forEach((term, i) => index.set(term, i));
  return { raw: m, index };
}

/** TF-IDF vectorisation identical to sklearn's TfidfVectorizer.transform for this config. */
export function vectorize(model: Model, text: string): SparseVector {
  const counts = new Map<number, number>();
  for (const gram of analyze(text)) {
    const i = model.index.get(gram);
    if (i !== undefined) counts.set(i, (counts.get(i) ?? 0) + 1);
  }
  const vec: SparseVector = new Map();
  let sumSq = 0;
  for (const [i, tf] of counts) {
    const t = model.raw.tfidf.sublinearTf ? 1 + Math.log(tf) : tf;
    const w = t * model.raw.idf[i]!;
    vec.set(i, w);
    sumSq += w * w;
  }
  const norm = Math.sqrt(sumSq);
  if (norm > 0) for (const [i, w] of vec) vec.set(i, w / norm);
  return vec;
}

export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** Scores a message with the same decision rule as training: spam iff P(spam) >= threshold. */
export function score(model: Model, text: string): Score {
  const vec = vectorize(model, text);
  const { coef, intercept, vocabulary, threshold } = model.raw;
  let logit = intercept;
  const contributions: FeatureContribution[] = [];
  for (const [i, w] of vec) {
    const c = coef[i]!;
    logit += w * c;
    contributions.push({ term: vocabulary[i]!, weight: w, coef: c, contribution: w * c });
  }
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const probability = sigmoid(logit);
  const isSpam = probability >= threshold;
  return {
    label: isSpam ? "spam" : "ham",
    isSpam,
    probability,
    confidence: isSpam ? probability : 1 - probability,
    logit,
    contributions,
  };
}
