import { describe, expect, it } from "vitest";
import { score, sigmoid } from "../src/model";
import { shippedModel } from "./helpers";

const model = shippedModel();

// Verbatim messages from the SMS Spam Collection's held-out 20% test split
// (see tests/fixtures/parity.json), so the model never saw them during training.
const REAL_SPAM = [
  "URGENT! We are trying to contact U. Todays draw shows that you have won a £800 prize GUARANTEED. Call 09050003091 from land line. Claim C52. Valid 12hrs only",
  "FREE entry into our £250 weekly comp just send the word ENTER to 84128 NOW. 18 T&C www.textcomp.com cust care 08712405020.",
  "Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B",
  "Do you want 750 anytime any network mins 150 text and a NEW video phone for only five pounds per week call 08000776320 now or reply for delivery Tomorrow",
];
const REAL_HAM = [
  "Where are you?when wil you reach here?",
  "I don't think I can get away for a trek that long with family in town, sorry",
  "Hey.. Something came up last min.. Think i wun be signing up tmr.. Hee",
  "Remind me how to get there and I shall do so",
];

describe("score (against docs/model.json)", () => {
  it.each(REAL_SPAM)("classifies real spam as spam: %s", (text) => {
    const s = score(model, text);
    expect(s.label).toBe("spam");
    expect(s.probability).toBeGreaterThanOrEqual(model.raw.threshold);
  });

  it.each(REAL_HAM)("classifies real ham as legitimate: %s", (text) => {
    const s = score(model, text);
    expect(s.label).toBe("ham");
    expect(s.probability).toBeLessThan(model.raw.threshold);
  });

  it("reports confidence as the probability of the predicted label", () => {
    const spam = score(model, REAL_SPAM[0]!);
    expect(spam.confidence).toBeCloseTo(spam.probability, 12);
    const ham = score(model, REAL_HAM[0]!);
    expect(ham.confidence).toBeCloseTo(1 - ham.probability, 12);
  });

  it("logit equals intercept plus the sum of feature contributions", () => {
    const s = score(model, REAL_SPAM[1]!);
    const sum = s.contributions.reduce((acc, c) => acc + c.contribution, model.raw.intercept);
    expect(s.logit).toBeCloseTo(sum, 10);
    expect(s.probability).toBeCloseTo(sigmoid(s.logit), 12);
  });

  it("scores empty input from the intercept alone", () => {
    const s = score(model, "");
    expect(s.logit).toBe(model.raw.intercept);
    expect(s.contributions).toEqual([]);
  });
});

describe("sigmoid", () => {
  it("is numerically stable at extremes", () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(1000)).toBe(1);
    expect(sigmoid(-1000)).toBe(0);
    expect(sigmoid(2) + sigmoid(-2)).toBeCloseTo(1, 15);
  });
});
