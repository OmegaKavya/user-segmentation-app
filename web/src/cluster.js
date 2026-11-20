// Cluster validity, in the browser.
//
// A direct port of analysis/validity.py. The two share one trivially
// reproducible generator, so they draw the same random numbers and agree
// exactly rather than approximately; scripts/verify_cluster.mjs checks that.

export function mulberry32(a) {
  return function next() {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const integers = (rng, n) => Math.floor(rng() * n);

function permutation(rng, values) {
  const out = values.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = integers(rng, i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function chooseWeighted(rng, weights) {
  let sum = 0;
  for (const w of weights) sum += w;
  const target = rng() * sum;
  let total = 0;
  for (let i = 0; i < weights.length; i += 1) {
    total += weights[i];
    if (total >= target) return i;
  }
  return weights.length - 1;
}

const sqdist = (a, b) => {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) { const d = a[i] - b[i]; total += d * d; }
  return total;
};

/** Lloyd's algorithm with k-means++ seeding. */
export function kmeans(X, k, seed = 42, iters = 60) {
  const rng = mulberry32(seed);
  const n = X.length;

  const centres = [X[integers(rng, n)].slice()];
  for (let c = 1; c < k; c += 1) {
    const d2 = X.map((p) => Math.min(...centres.map((q) => sqdist(p, q))));
    const total = d2.reduce((a, b) => a + b, 0);
    centres.push((total <= 0 ? X[integers(rng, n)] : X[chooseWeighted(rng, d2)]).slice());
  }

  let labels = new Int32Array(n);
  for (let it = 0; it < iters; it += 1) {
    let moved = false;
    for (let i = 0; i < n; i += 1) {
      let best = 0, bestD = Infinity;
      for (let j = 0; j < k; j += 1) {
        const d = sqdist(X[i], centres[j]);
        if (d < bestD) { bestD = d; best = j; }
      }
      if (labels[i] !== best) { labels[i] = best; moved = true; }
    }
    if (!moved && it > 0) break;
    for (let j = 0; j < k; j += 1) {
      const members = [];
      for (let i = 0; i < n; i += 1) if (labels[i] === j) members.push(X[i]);
      if (!members.length) continue;
      const mean = new Array(X[0].length).fill(0);
      for (const m of members) for (let f = 0; f < mean.length; f += 1) mean[f] += m[f];
      for (let f = 0; f < mean.length; f += 1) mean[f] /= members.length;
      centres[j] = mean;
    }
  }

  let inertia = 0;
  for (let i = 0; i < n; i += 1) inertia += sqdist(X[i], centres[labels[i]]);
  return { labels, inertia, centres };
}

/** Mean silhouette, on a subsample when the data is large. */
export function silhouette(X, labels, sample = 400, seed = 42) {
  const rng = mulberry32(seed);
  const n = X.length;
  const idx = n > sample
    ? permutation(rng, Array.from({ length: n }, (_, i) => i)).slice(0, sample)
    : Array.from({ length: n }, (_, i) => i);

  const pts = idx.map((i) => X[i]);
  const ls = idx.map((i) => labels[i]);
  const groups = [...new Set(ls)];
  if (groups.length < 2) return 0;

  let total = 0;
  for (let i = 0; i < pts.length; i += 1) {
    const sums = new Map(groups.map((g) => [g, 0]));
    const counts = new Map(groups.map((g) => [g, 0]));
    for (let j = 0; j < pts.length; j += 1) {
      if (i === j) continue;
      const d = Math.sqrt(sqdist(pts[i], pts[j]));
      sums.set(ls[j], sums.get(ls[j]) + d);
      counts.set(ls[j], counts.get(ls[j]) + 1);
    }
    if (counts.get(ls[i]) === 0) continue;
    const a = sums.get(ls[i]) / counts.get(ls[i]);
    let b = Infinity;
    for (const g of groups) {
      if (g === ls[i] || counts.get(g) === 0) continue;
      b = Math.min(b, sums.get(g) / counts.get(g));
    }
    total += (b - a) / Math.max(a, b);
  }
  return total / pts.length;
}

/** Each column permuted on its own: marginals kept, joint structure gone. */
export function shuffled(X, rng) {
  const n = X.length, d = X[0].length;
  const out = Array.from({ length: n }, () => new Array(d));
  for (let f = 0; f < d; f += 1) {
    const col = permutation(rng, X.map((row) => row[f]));
    for (let i = 0; i < n; i += 1) out[i][f] = col[i];
  }
  return out;
}

/** The whole assessment: per k, plus one verdict across all of them.
 *
 * Testing seven values of k is seven chances to clear a threshold. The
 * max-statistic permutation test handles that by construction: each shuffled
 * copy is scored at every k and only its best is kept, so the null is the
 * distribution of "the best silhouette anyone could get by searching over k".
 * No correction is needed afterwards, and none is possible anyway, since with
 * R shuffles the smallest attainable p is 1/(R+1).
 */
export function assess(X, { ks = [2, 3, 4, 5, 6, 7, 8], repeats = 30, seed = 42, onProgress } = {}) {
  const rng = mulberry32(seed);
  const steps = ks.length * (repeats + 1);
  let done = 0;

  const perK = ks.map((k) => {
    const { labels, inertia } = kmeans(X, k, seed);
    const sil = silhouette(X, labels, 400, seed);
    const sizes = [];
    for (let j = 0; j < k; j += 1) sizes.push(labels.reduce((a, l) => a + (l === j ? 1 : 0), 0));
    done += 1; onProgress?.(done / steps);
    return { k, inertia, silhouette: sil, sizes };
  });

  const nullByK = ks.map(() => []);
  const nullBest = [];
  for (let r = 0; r < repeats; r += 1) {
    const Z = shuffled(X, rng);
    let best = -Infinity;
    ks.forEach((k, i) => {
      const s = silhouette(Z, kmeans(Z, k, seed + r).labels, 400, seed);
      nullByK[i].push(s);
      if (s > best) best = s;
      done += 1; onProgress?.(done / steps);
    });
    nullBest.push(best);
  }

  perK.forEach((row, i) => {
    const nulls = nullByK[i];
    const mean = nulls.reduce((a, b) => a + b, 0) / nulls.length;
    const sd = Math.sqrt(nulls.reduce((a, b) => a + (b - mean) ** 2, 0) / nulls.length);
    row.nullMean = mean;
    row.nullSd = sd;
    row.z = sd > 0 ? (row.silhouette - mean) / sd : 0;
  });

  const best = perK.reduce((a, b) => (b.silhouette > a.silhouette ? b : a));
  const beaten = nullBest.filter((v) => v >= best.silhouette).length;
  return {
    perK,
    verdict: {
      bestK: best.k,
      bestSilhouette: best.silhouette,
      nullBestMean: nullBest.reduce((a, b) => a + b, 0) / nullBest.length,
      p: (1 + beaten) / (1 + repeats),
      repeats,
    },
  };
}
