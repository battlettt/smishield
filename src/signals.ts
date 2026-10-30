import { score, type FeatureContribution, type Model, type Score } from "./model";
import { CURRENCY_TOKEN, LONG_NUMBER_TOKEN, SHORT_URL_TOKEN, URL_TOKEN, findUrls } from "./tokenizer";

export type SignalId =
  | "shortened-url"
  | "link"
  | "urgency"
  | "prize-money"
  | "account-security"
  | "premium-number"
  | "suspicious-sender";

/** A human-readable warning sign found by pattern matching on the raw message. */
export interface Signal {
  id: SignalId;
  label: string;
  detail: string;
  evidence: string[];
}

export interface Explanation {
  score: Score;
  signals: Signal[];
  spamTerms: FeatureContribution[]; // strongest model features pushing towards spam
  hamTerms: FeatureContribution[]; // strongest model features pushing towards legitimate
  summary: string;
}

const URGENCY_PHRASES = [
  "act now", "limited time", "urgent", "urgently", "immediately", "right away", "asap",
  "expires", "expiring", "expire", "expired", "final notice", "final attempt", "last chance",
  "today only", "hurry", "don't miss", "dont miss", "now or never", "respond now", "call now",
  "reply now", "claim now", "click now", "action required", "deadline", "before midnight",
  "ends tonight", "ends today",
];
const URGENCY_PATTERNS = [
  /within [0-9]+ ?(?:hours?|hrs?|mins?|minutes?|days?)(?![a-z])/g,
  /valid (?:for )?(?:only )?[0-9]+ ?(?:hours?|hrs?|mins?|minutes?|days?)(?![a-z])/g,
];

const PRIZE_PHRASES = [
  "won", "win", "winner", "winning", "prize", "claim", "cash", "reward", "free", "bonus",
  "gift card", "giftcard", "voucher", "award", "jackpot", "lottery", "congratulations",
  "congrats", "selected", "guaranteed", "refund", "rebate", "loan", "cashback",
];
const MONEY_PATTERNS = [/[£$€] ?[0-9][0-9,.]*/g, /[0-9][0-9,.]* ?(?:pounds|dollars|euros|usd|gbp)/g];

const ACCOUNT_PHRASES = [
  "verify", "verification", "account", "suspended", "locked", "blocked", "password", "pin",
  "login", "log in", "sign in", "bank", "unpaid", "overdue", "payment", "confirm your",
  "update your", "security alert", "unusual activity", "delivery", "parcel", "package",
  "redelivery", "customs fee", "toll", "tax",
];

const PREMIUM_PATTERNS = [
  /(?<![0-9])09[0-9]{8,9}(?![0-9])/g, // UK premium-rate numbers (0906..., 0871...)
  /(?<![0-9])087[0-9]{7,8}(?![0-9])/g,
  /(?:text|txt|send|reply|sms)(?: [a-z0-9]+){0,3} to [0-9]{4,6}(?![0-9])/g, // "text WIN to 80086"
  /[0-9]+p ?(?:\/|per) ?(?:msg|min|text|sms|wk|week)/g, // "150p/msg"
];

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findPhrases(lower: string, phrases: readonly string[]): string[] {
  const found: string[] = [];
  for (const p of phrases) {
    const re = new RegExp(`(?<![a-z])${escapeRegExp(p)}(?![a-z])`);
    if (re.test(lower)) found.push(p);
  }
  return found;
}

function findPatterns(lower: string, patterns: readonly RegExp[]): string[] {
  const found = new Set<string>();
  for (const re of patterns) for (const m of lower.matchAll(re)) found.add(m[0].trim());
  return [...found];
}

/** Flags sender IDs typical of bulk spam: email-to-SMS gateways, short codes,
 *  odd-length numbers and look-alike brand names (digits standing in for letters). */
export function senderSignal(sender: string): Signal | null {
  const s = sender.trim();
  if (!s) return null;
  const evidence: string[] = [];
  const digits = s.replace(/[^0-9]/g, "");
  const isNumeric = /^\+?[0-9 ()-]+$/.test(s);
  if (/^[^@\s]+@[^@\s]+\.[a-z]{2,}$/i.test(s)) {
    evidence.push("sent from an email address (email-to-SMS gateways are common for bulk spam)");
  } else if (isNumeric && digits.length >= 4 && digits.length <= 6) {
    evidence.push(`short code ${digits} (used for bulk and premium-rate messaging)`);
  } else if (isNumeric && (digits.length > 15 || (digits.length > 0 && digits.length < 4))) {
    evidence.push(`number has an unusual length (${digits.length} digits)`);
  } else if (!isNumeric && /[a-z][0-9]|[0-9][a-z]/i.test(s) && /[01345]/.test(s)) {
    evidence.push(`"${s}" mixes letters and look-alike digits, as in brand impersonation (e.g. AMAZ0N)`);
  }
  if (evidence.length === 0) return null;
  return {
    id: "suspicious-sender",
    label: "Suspicious sender pattern",
    detail: evidence.join("; "),
    evidence: [s],
  };
}

/** Pattern-based warning signs, independent of the statistical model. */
export function detectSignals(text: string, sender = ""): Signal[] {
  const lower = text.toLowerCase();
  const signals: Signal[] = [];

  const urls = findUrls(text);
  const short = urls.filter((u) => u.shortened);
  if (short.length > 0) {
    signals.push({
      id: "shortened-url",
      label: "Shortened URL",
      detail: `link shortener (${[...new Set(short.map((u) => u.domain))].join(", ")}) hides the real destination`,
      evidence: short.map((u) => u.url),
    });
  }
  const plain = urls.filter((u) => !u.shortened);
  if (plain.length > 0) {
    signals.push({
      id: "link",
      label: "Contains a link",
      detail: `links to ${[...new Set(plain.map((u) => u.domain))].join(", ")}; check it before tapping`,
      evidence: plain.map((u) => u.url),
    });
  }

  const urgency = [...findPhrases(lower, URGENCY_PHRASES), ...findPatterns(lower, URGENCY_PATTERNS)];
  if (urgency.length > 0) {
    signals.push({
      id: "urgency",
      label: "Urgency language",
      detail: `pressure to act quickly: ${quoteList(urgency)}`,
      evidence: urgency,
    });
  }

  const prize = [...findPhrases(lower, PRIZE_PHRASES), ...findPatterns(lower, MONEY_PATTERNS)];
  if (prize.length > 0) {
    signals.push({
      id: "prize-money",
      label: "Prize / money language",
      detail: `promises of money or prizes: ${quoteList(prize)}`,
      evidence: prize,
    });
  }

  const account = findPhrases(lower, ACCOUNT_PHRASES);
  if (account.length > 0) {
    signals.push({
      id: "account-security",
      label: "Account / payment / delivery language",
      detail: `common smishing lures: ${quoteList(account)}`,
      evidence: account,
    });
  }

  const premium = findPatterns(lower, PREMIUM_PATTERNS);
  if (premium.length > 0) {
    signals.push({
      id: "premium-number",
      label: "Premium-rate number or short code",
      detail: `asks you to call or text a paid or bulk-messaging number: ${quoteList(premium)}`,
      evidence: premium,
    });
  }

  const senderSig = senderSignal(sender);
  if (senderSig) signals.push(senderSig);

  return signals;
}

function quoteList(items: string[], max = 4): string {
  const uniq = [...new Set(items)];
  const shown = uniq.slice(0, max).map((s) => `"${s}"`);
  return uniq.length > max ? `${shown.join(", ")} and ${uniq.length - max} more` : shown.join(", ");
}

/** Renders a model feature (which may contain placeholder tokens) for humans. */
export function describeTerm(term: string): string {
  return term
    .split(" ")
    .map((t) => {
      switch (t) {
        case URL_TOKEN: return "[link]";
        case SHORT_URL_TOKEN: return "[short link]";
        case CURRENCY_TOKEN: return "[£/$/€]";
        case LONG_NUMBER_TOKEN: return "[phone number/short code]";
        default: return t;
      }
    })
    .join(" ");
}

/** Percentage for display; never rounds up to a claim of certainty. */
export function formatConfidence(p: number): string {
  const r = Math.round(p * 100);
  return r >= 100 ? ">99%" : `${r}%`;
}

function joinNatural(parts: string[]): string {
  if (parts.length <= 1) return parts.join("");
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

/** Scores a message and explains the verdict in plain English. */
export function explain(model: Model, text: string, sender = ""): Explanation {
  const s = score(model, text);
  const signals = detectSignals(text, sender);
  const spamTerms = s.contributions.filter((c) => c.contribution > 0).slice(0, 5);
  const hamTerms = s.contributions.filter((c) => c.contribution < 0).slice(0, 5);
  const pct = formatConfidence(s.confidence);

  const termText = (terms: FeatureContribution[]) =>
    terms.slice(0, 4).map((t) => `"${describeTerm(t.term)}"`).join(", ");
  const signalText = signals.map((sig) => sig.label.charAt(0).toLowerCase() + sig.label.slice(1));

  let summary: string;
  if (s.isSpam) {
    const reasons: string[] = [...signalText];
    if (spamTerms.length > 0) reasons.push(`wording the model associates with spam (${termText(spamTerms)})`);
    summary = `Likely spam/smishing (${pct} confidence). ` +
      (reasons.length > 0 ? `Main signals: ${joinNatural(reasons)}.` : "");
  } else {
    summary = `Likely legitimate (${pct} confidence). ` +
      (hamTerms.length > 0
        ? `The wording resembles ordinary personal messages (${termText(hamTerms)}).`
        : "The model found little that resembles known spam.");
    if (signals.length > 0) {
      summary += ` However, some warning signs are present (${joinNatural(signalText)}); ` +
        "the model's score stayed below the spam threshold, so still be careful with links and requests.";
    }
  }

  return { score: s, signals, spamTerms, hamTerms, summary: summary.trim() };
}
