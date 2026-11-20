// The assessment runs here, not on the main thread.
//
// Twenty shuffled nulls at each of seven values of k is 147 clusterings plus
// 147 silhouettes. That is a second or two of solid arithmetic, which would
// freeze the page if it ran where the page is drawn.
import { assess } from "./cluster.js";

self.onmessage = (event) => {
  const { X, ks, repeats } = event.data;
  try {
    const results = assess(X, {
      ks, repeats,
      onProgress: (fraction) => self.postMessage({ type: "progress", fraction }),
    });
    self.postMessage({ type: "done", results });
  } catch (error) {
    self.postMessage({ type: "error", message: String(error?.message || error) });
  }
};
