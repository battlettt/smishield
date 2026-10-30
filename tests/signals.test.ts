import { describe, expect, it } from "vitest";
import { describeTerm, detectSignals, explain, formatConfidence, senderSignal } from "../src/signals";
import { shippedModel } from "./helpers";

const model = shippedModel();
const SMISH =
  "URGENT: your parcel is on hold. Act now, limited time to pay the redelivery fee: https://bit.ly/3Pq9xZ";

describe("detectSignals", () => {
  it("surfaces a shortened URL and urgency language", () => {
    const signals = detectSignals(SMISH);
    const ids = signals.map((s) => s.id);
    expect(ids).toContain("shortened-url");
    expect(ids).toContain("urgency");
    expect(ids).not.toContain("link"); // the only link is the shortened one

    const short = signals.find((s) => s.id === "shortened-url")!;
    expect(short.detail).toContain("bit.ly");
    const urgency = signals.find((s) => s.id === "urgency")!;
    expect(urgency.evidence).toEqual(expect.arrayContaining(["urgent", "act now", "limited time"]));
  });

  it("detects prize/money language and premium-rate numbers", () => {
    const signals = detectSignals(
      "Congratulations! You have WON a £1000 prize. To claim call 09050003091 or text WIN to 80086",
    );
    const prize = signals.find((s) => s.id === "prize-money")!;
    expect(prize.evidence).toEqual(expect.arrayContaining(["won", "prize", "claim", "£1000"]));
    const premium = signals.find((s) => s.id === "premium-number")!;
    expect(premium.evidence).toEqual(expect.arrayContaining(["09050003091", "text win to 80086"]));
  });

  it("detects expiry windows as urgency", () => {
    const urgency = detectSignals("Claim C52. Valid 12hrs only").find((s) => s.id === "urgency")!;
    expect(urgency.evidence).toContain("valid 12hrs");
  });

  it("flags account/payment lures and a plain link", () => {
    const ids = detectSignals("Your bank account is suspended. Verify at secure-login.example.com").map((s) => s.id);
    expect(ids).toEqual(expect.arrayContaining(["account-security", "link"]));
  });

  it("matches whole words only", () => {
    expect(detectSignals("I went to the window, pinned it and freed the cat").map((s) => s.id)).toEqual([]);
  });

  it("returns nothing for an ordinary message", () => {
    expect(detectSignals("Running 10 mins late, see you at the pub")).toEqual([]);
  });
});

describe("senderSignal", () => {
  it("flags suspicious sender patterns", () => {
    expect(senderSignal("promo@mailer-sms.biz")?.id).toBe("suspicious-sender");
    expect(senderSignal("80086")?.detail).toContain("short code");
    expect(senderSignal("AMAZ0N")?.detail).toContain("look-alike");
  });

  it("does not flag ordinary senders", () => {
    expect(senderSignal("")).toBeNull();
    expect(senderSignal("+44 7700 900123")).toBeNull();
    expect(senderSignal("Mum")).toBeNull();
  });
});

describe("explain (against docs/model.json)", () => {
  it("names the shortened URL and urgency signals in the summary of a smishing message", () => {
    const e = explain(model, SMISH);
    expect(e.score.isSpam).toBe(true);
    expect(e.summary).toMatch(/^Likely spam\/smishing \((?:\d+|>99)% confidence\)/);
    expect(e.summary).toContain("shortened URL");
    expect(e.summary).toContain("urgency language");
    expect(e.spamTerms.length).toBeGreaterThan(0);
    expect(e.spamTerms.every((t) => t.contribution > 0)).toBe(true);
  });

  it("includes the sender signal when a sender is given", () => {
    const e = explain(model, SMISH, "80086");
    expect(e.signals.map((s) => s.id)).toContain("suspicious-sender");
    expect(e.summary).toContain("suspicious sender pattern");
  });

  it("explains a legitimate message with ham-leaning words", () => {
    const e = explain(model, "Hey, are we still ok for dinner tonight? I'll bring the wine");
    expect(e.score.isSpam).toBe(false);
    expect(e.summary).toMatch(/^Likely legitimate \((?:\d+|>99)% confidence\)/);
    expect(e.hamTerms.every((t) => t.contribution < 0)).toBe(true);
  });

  it("never displays a rounded 100% confidence", () => {
    expect(formatConfidence(0.9996)).toBe(">99%");
    expect(formatConfidence(0.994)).toBe("99%");
    expect(formatConfidence(0.5)).toBe("50%");
  });

  it("renders placeholder tokens readably", () => {
    expect(describeTerm("call __longnum__")).toBe("call [phone number/short code]");
    expect(describeTerm("__currency__")).toBe("[£/$/€]");
  });
});
