// Tokenizer mirrored exactly from train.py. Any change here must be made there too
// (and the model retrained); tests/parity.test.ts checks both agree on real messages.

export const URL_TOKEN = "__url__";
export const SHORT_URL_TOKEN = "__shorturl__";
export const CURRENCY_TOKEN = "__currency__";
export const LONG_NUMBER_TOKEN = "__longnum__";
export const LONG_NUMBER_MIN_DIGITS = 5;

export const SHORTENER_DOMAINS: readonly string[] = [
  "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
  "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "bl.ink",
  "t.ly", "s.id", "v.gd", "short.io", "tiny.one", "lnkd.in", "qr.ae",
];

// Explicit character classes (no \s, \b, \d) so JavaScript and Python agree.
const NON_SPACE = "[^ \\t\\n\\r\\f\\v]";
const TLDS =
  "(?:com|net|org|info|biz|co|uk|ly|gl|gd|cc|gy|at|io|me|link|ink|id|in|ae|one|us|xyz|top|ru|cn)";
const URL_PATTERN =
  `(?:https?://|www\\.)${NON_SPACE}+` +
  `|(?<![a-z0-9.-])(?:[a-z0-9-]+\\.)+${TLDS}(?![a-z0-9-])(?:/${NON_SPACE}*)?`;
const TOKEN_SOURCE = `(${URL_PATTERN})|([£$€])|([a-z0-9]+)`;
const DIGITS_RE = /^[0-9]+$/;

/** Extracts the bare host (no scheme, no www., no path) from a lowercased URL match. */
export function urlDomain(url: string): string {
  let d = url.replace(/^https?:\/\//, "").replace(/^www\./, "");
  d = d.split("/")[0]!.split("?")[0]!;
  return d.replace(/[^a-z0-9]+$/, "");
}

export function isShortenerDomain(domain: string): boolean {
  return SHORTENER_DOMAINS.includes(domain);
}

/** Lowercases and splits a message into word tokens, replacing URLs, currency
 *  symbols and long digit runs (phone numbers, short codes) with placeholder tokens. */
export function tokenize(text: string): string[] {
  const tokens: string[] = [];
  const re = new RegExp(TOKEN_SOURCE, "g");
  for (const m of text.toLowerCase().matchAll(re)) {
    const [, url, currency, word] = m;
    if (url !== undefined) {
      tokens.push(URL_TOKEN);
      if (isShortenerDomain(urlDomain(url))) tokens.push(SHORT_URL_TOKEN);
    } else if (currency !== undefined) {
      tokens.push(CURRENCY_TOKEN);
    } else if (word !== undefined) {
      tokens.push(
        DIGITS_RE.test(word) && word.length >= LONG_NUMBER_MIN_DIGITS ? LONG_NUMBER_TOKEN : word,
      );
    }
  }
  return tokens;
}

/** Unigrams followed by space-joined bigrams, matching sklearn's ngram_range=(1, 2). */
export function analyze(text: string): string[] {
  const toks = tokenize(text);
  const grams = [...toks];
  for (let i = 0; i + 1 < toks.length; i++) grams.push(`${toks[i]} ${toks[i + 1]}`);
  return grams;
}

/** Finds URLs in the raw message (used for explanations, not for scoring). */
export function findUrls(text: string): { url: string; domain: string; shortened: boolean }[] {
  const re = new RegExp(URL_PATTERN, "g");
  return [...text.toLowerCase().matchAll(re)].map((m) => {
    const domain = urlDomain(m[0]);
    return { url: m[0], domain, shortened: isShortenerDomain(domain) };
  });
}
