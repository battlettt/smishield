import { describe, expect, it } from "vitest";
import {
  CURRENCY_TOKEN, LONG_NUMBER_TOKEN, SHORT_URL_TOKEN, SHORTENER_DOMAINS, URL_TOKEN,
  analyze, findUrls, tokenize,
} from "../src/tokenizer";
import { readJson } from "./helpers";
import type { ModelJson } from "../src/model";

describe("tokenize", () => {
  it("lowercases", () => {
    expect(tokenize("FREE Entry NOW")).toEqual(["free", "entry", "now"]);
  });

  it("splits on punctuation and drops it", () => {
    expect(tokenize("Ok lar... Joking wif u oni...")).toEqual(["ok", "lar", "joking", "wif", "u", "oni"]);
    expect(tokenize("don't! call-me, (please)?")).toEqual(["don", "t", "call", "me", "please"]);
  });

  it("returns no tokens for empty or punctuation-only input", () => {
    expect(tokenize("")).toEqual([]);
    expect(tokenize("... !!! ???")).toEqual([]);
  });

  it("keeps mixed alphanumerics but replaces long digit runs", () => {
    expect(tokenize("txt WIN to 87121 for 150p, 2nite")).toEqual(
      ["txt", "win", "to", LONG_NUMBER_TOKEN, "for", "150p", "2nite"],
    );
    expect(tokenize("call 09061701461 or 1234")).toEqual(["call", LONG_NUMBER_TOKEN, "or", "1234"]);
  });

  it("maps currency symbols to a token", () => {
    expect(tokenize("won £1000 or $50")).toEqual(["won", CURRENCY_TOKEN, "1000", "or", CURRENCY_TOKEN, "50"]);
  });

  it("replaces full URLs with a single URL token", () => {
    expect(tokenize("see https://example.com/a/b?c=d now")).toEqual(["see", URL_TOKEN, "now"]);
    expect(tokenize("goto www.comuk.net login")).toEqual(["goto", URL_TOKEN, "login"]);
  });

  it("recognises bare domains, including ones with paths", () => {
    expect(tokenize("visit getzed.co.uk today")).toEqual(["visit", URL_TOKEN, "today"]);
    expect(tokenize("PocketBabe.co.uk/pics")).toEqual([URL_TOKEN]);
  });

  it("adds a short-URL token for known link shorteners", () => {
    expect(tokenize("Go to bit.ly/3xYz now")).toEqual(["go", "to", URL_TOKEN, SHORT_URL_TOKEN, "now"]);
    expect(tokenize("https://tinyurl.com/abc")).toEqual([URL_TOKEN, SHORT_URL_TOKEN]);
    expect(tokenize("https://example.com/bit.ly")).toEqual([URL_TOKEN]);
  });

  it("does not treat ordinary sentence breaks as URLs", () => {
    expect(tokenize("I'm at home.Call me")).toEqual(["i", "m", "at", "home", "call", "me"]);
  });
});

describe("analyze", () => {
  it("emits unigrams then space-joined bigrams", () => {
    expect(analyze("Win cash now")).toEqual(["win", "cash", "now", "win cash", "cash now"]);
  });
});

describe("findUrls", () => {
  it("extracts domains and flags shorteners", () => {
    const urls = findUrls("Track: https://bit.ly/abc. Or www.royalmail.com/track.");
    expect(urls.map((u) => [u.domain, u.shortened])).toEqual([["bit.ly", true], ["royalmail.com", false]]);
  });
});

describe("tokenizer config matches the shipped model", () => {
  it("uses the same placeholder tokens and shortener list as train.py", () => {
    const m = readJson<ModelJson>("docs/model.json");
    expect(m.tokenizer.urlToken).toBe(URL_TOKEN);
    expect(m.tokenizer.shortUrlToken).toBe(SHORT_URL_TOKEN);
    expect(m.tokenizer.currencyToken).toBe(CURRENCY_TOKEN);
    expect(m.tokenizer.longNumberToken).toBe(LONG_NUMBER_TOKEN);
    expect(m.tokenizer.shortenerDomains).toEqual([...SHORTENER_DOMAINS]);
    expect(m.tokenizer.ngramRange).toEqual([1, 2]);
  });
});
