import { describe, expect, it } from "vitest";
import { score } from "../src/model";
import { tokenize } from "../src/tokenizer";
import { readJson, shippedModel } from "./helpers";

interface ParityCase {
  text: string;
  label: "spam" | "ham";
  tokens: string[];
  probability: number;
  predicted: "spam" | "ham";
}

const model = shippedModel();
const fixture = readJson<{ threshold: number; cases: ParityCase[] }>("tests/fixtures/parity.json");

// Each case was produced by train.py with scikit-learn on held-out test messages.
describe("TypeScript reproduces training-time output on held-out messages", () => {
  it("has a meaningful number of cases from both classes", () => {
    expect(fixture.cases.filter((c) => c.label === "spam").length).toBeGreaterThanOrEqual(30);
    expect(fixture.cases.filter((c) => c.label === "ham").length).toBeGreaterThanOrEqual(30);
    expect(fixture.threshold).toBe(model.raw.threshold);
  });

  it("tokenizes identically to the Python tokenizer", () => {
    for (const c of fixture.cases) expect(tokenize(c.text), c.text).toEqual(c.tokens);
  });

  it("matches scikit-learn's probabilities and decisions", () => {
    for (const c of fixture.cases) {
      const s = score(model, c.text);
      expect(Math.abs(s.probability - c.probability), c.text).toBeLessThan(1e-9);
      expect(s.label, c.text).toBe(c.predicted);
    }
  });
});
