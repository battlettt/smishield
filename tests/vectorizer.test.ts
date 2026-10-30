import { describe, expect, it } from "vitest";
import { vectorize } from "../src/model";
import { LONG_NUMBER_TOKEN } from "../src/tokenizer";
import { shippedModel } from "./helpers";

const model = shippedModel();
const idx = (term: string): number => {
  const i = model.index.get(term);
  if (i === undefined) throw new Error(`"${term}" not in shipped vocabulary`);
  return i;
};

describe("vectorize (against docs/model.json)", () => {
  it("has a consistent, capped vocabulary", () => {
    expect(model.raw.vocabulary.length).toBeGreaterThan(0);
    expect(model.raw.vocabulary.length).toBeLessThanOrEqual(3000);
    expect(model.raw.idf.length).toBe(model.raw.vocabulary.length);
    expect(model.raw.coef.length).toBe(model.raw.vocabulary.length);
    expect(new Set(model.raw.vocabulary).size).toBe(model.raw.vocabulary.length);
  });

  it("maps known unigrams and bigrams to their vocabulary indices", () => {
    const v = vectorize(model, "Call 09061701461 to claim your prize");
    for (const term of ["call", LONG_NUMBER_TOKEN, "claim", "prize", `call ${LONG_NUMBER_TOKEN}`, "to claim", "claim your"]) {
      expect(v.has(idx(term)), term).toBe(true);
    }
  });

  it("ignores out-of-vocabulary tokens", () => {
    expect(vectorize(model, "zzqxv blorptastic").size).toBe(0);
  });

  it("produces an l2-normalised vector", () => {
    const v = vectorize(model, "FREE entry to win cash! Text WIN to 80086 now");
    const norm = Math.sqrt([...v.values()].reduce((s, w) => s + w * w, 0));
    expect(norm).toBeCloseTo(1, 12);
  });

  it("weights a single known token by sublinear tf * idf before normalising", () => {
    const one = vectorize(model, "prize");
    expect([...one.entries()]).toEqual([[idx("prize"), 1]]);
    // With two distinct tokens, the ratio of weights equals the ratio of their idf values.
    const two = vectorize(model, "prize prize prize prize claim");
    const ratio = two.get(idx("prize"))! / two.get(idx("claim"))!;
    const expected = ((1 + Math.log(4)) * model.raw.idf[idx("prize")]!) / model.raw.idf[idx("claim")]!;
    expect(ratio).toBeCloseTo(expected, 10);
  });
});
