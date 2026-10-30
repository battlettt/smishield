import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadModel, type Model } from "../src/model";

const root = fileURLToPath(new URL("..", import.meta.url));

export function readJson<T = unknown>(rel: string): T {
  return JSON.parse(readFileSync(root + rel, "utf8")) as T;
}

/** The shipped artifact, loaded exactly as the demo page loads it. */
export function shippedModel(): Model {
  return loadModel(readJson("docs/model.json"));
}
