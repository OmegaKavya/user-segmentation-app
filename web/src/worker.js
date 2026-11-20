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

    post("stage", { stage: "permutation" });
    post("permutation", { result: assess(X, { ks, repeats }) });

    const S = subsample(X, HEAVY_LIMIT);
    post("stage", { stage: "gap", sampled: S.length < X.length ? S.length : null });
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
