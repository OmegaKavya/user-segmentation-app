// Every test runs here, and reports as each one finishes.
//
// Four analyses over seven values of k is a lot of arithmetic. Posting each
// result as it lands means the page fills in progressively instead of showing
// a spinner for twenty seconds, and the permutation test, which is the one
// that answers the headline question, is deliberately run first.
import { assess, mulberry32 } from "./cluster.js";
import { gapStatistic, stability, byAlgorithm } from "./methods.js";

// The heavy tests are quadratic in the number of points. Five hundred is far
// more than enough to answer a yes or no, and keeps the whole run in seconds.
const HEAVY_LIMIT = 500;
// The permutation test is linear in the rows, since its silhouette samples
// 400, but it runs 217 clusterings: on a 100,000 row upload that is minutes
// of a stalled tab. Two thousand keeps both built-in datasets whole, so
// their results are exactly those the page has always shown.
const PERMUTATION_LIMIT = 2000;

function subsample(X, limit, seed = 42) {
  if (X.length <= limit) return X;
  const rng = mulberry32(seed);
  const idx = Array.from({ length: X.length }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, limit).sort((a, b) => a - b).map((i) => X[i]);
}

self.onmessage = (event) => {
  const { X, ks, repeats } = event.data;
  try {
    const post = (type, payload) => self.postMessage({ type, ...payload });

    // Same seed and shuffle, so every point in the smaller sample is in the larger.
    const P = subsample(X, PERMUTATION_LIMIT);
    const S = subsample(X, HEAVY_LIMIT);
    const size = (sample) => (sample.length < X.length ? sample.length : null);
    // Sent before the first result, so the verdict never appears to cover
    // rows it did not see.
    post("stage", { stage: "permutation", sampled: { permutation: size(P), evidence: size(S) } });
    post("permutation", { result: assess(P, { ks, repeats }) });

    post("stage", { stage: "gap" });
    post("gap", { result: gapStatistic(S, ks, 42, 10) });

    post("stage", { stage: "algorithms" });
    post("algorithms", { result: byAlgorithm(S, ks, 42) });

    post("stage", { stage: "stability" });
    const out = {};
    for (const algo of ["kmeans", "gmm", "ward"]) {
      out[algo] = stability(S, ks, 42, algo, 6);
      post("stability", { result: { ...out } });
    }

    post("done", {});
  } catch (error) {
    self.postMessage({ type: "error", message: String(error?.stack || error) });
  }
};
