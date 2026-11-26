import test from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import { assess, mulberry32 } from "./cluster.js";

// The worker talks to the page through self; here self records what it posts.
const posted = [];
globalThis.self = { postMessage: (m) => posted.push(m) };
await import("./worker.js");

const ks = [2, 3];
const repeats = 2;

function run(X) {
  posted.length = 0;
  self.onmessage({ data: { X, ks, repeats } });
  const error = posted.find((m) => m.type === "error");
  assert.ok(!error, error?.message);
  return posted.slice();
}

const clusteredRows = (n) => {
  const rng = mulberry32(7);
  return Array.from({ length: n }, (_, i) => [i % 3 + rng() * 0.3, (i % 3) * 2 + rng() * 0.3, rng()]);
};

test("a large upload is sampled before the permutation test, and the page is told first", () => {
  const messages = run(clusteredRows(2500));
  assert.deepStrictEqual(messages[0], { type: "stage", stage: "permutation", sampled: { permutation: 2000, evidence: 500 } });
  const { result } = messages.find((m) => m.type === "permutation");
  for (const row of result.perK) assert.strictEqual(row.sizes.reduce((a, b) => a + b, 0), 2000, `k=${row.k}`);
});

test("the built-in datasets are tested whole, so their verdicts are unchanged", () => {
  const datasets = JSON.parse(fs.readFileSync(new URL("./datasets.json", import.meta.url), "utf8"));
  for (const [name, { X }] of Object.entries(datasets)) {
    const messages = run(X);
    assert.strictEqual(messages[0].sampled.permutation, null, name);
    assert.deepStrictEqual(messages.find((m) => m.type === "permutation").result, assess(X, { ks, repeats }), name);
  }
});
