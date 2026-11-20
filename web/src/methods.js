// Three validity tests, ported from analysis/methods.py.
//
// They are deliberately different in kind. The permutation test asks whether
// the structure beats a null with none; the gap statistic asks whether the
// clusters are tighter than uniform spread would give; stability asks whether
// the same partition comes back when the data is resampled. Agreeing, they
// are strong evidence. Disagreeing, they say which assumption is load bearing.
import { mulberry32, kmeans, silhouette } from "./cluster.js";

const integers = (rng, n) => Math.floor(rng() * n);

function permutation(rng, values) {
  const out = values.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = integers(rng, i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// ---- algorithms -------------------------------------------------------

/** Expectation maximisation with diagonal covariances.
 *
 * Diagonal rather than full: a full covariance needs a matrix inverse per
 * component per iteration. The assumption is stated rather than hidden, and
 * what matters is retained, that components may have different widths along
 * each axis, so elongated groups that k-means splits can survive.
 */
export function gmmLabels(X, k, seed, iters = 40, reg = 1e-6) {
  const rng = mulberry32(seed);
  const n = X.length, d = X[0].length;

  const order = permutation(rng, Array.from({ length: n }, (_, i) => i));
  const means = order.slice(0, k).map((i) => X[i].slice());

  const colVar = new Array(d).fill(0);
  const colMean = new Array(d).fill(0);
  for (const row of X) for (let f = 0; f < d; f += 1) colMean[f] += row[f] / n;
  for (const row of X) for (let f = 0; f < d; f += 1) colVar[f] += (row[f] - colMean[f]) ** 2 / n;
  const variances = Array.from({ length: k }, () => colVar.map((v) => v + reg));
  let weights = new Array(k).fill(1 / k);

  const logp = Array.from({ length: n }, () => new Float64Array(k));
  for (let it = 0; it < iters; it += 1) {
    for (let j = 0; j < k; j += 1) {
      let logDet = 0;
      for (let f = 0; f < d; f += 1) logDet += Math.log(2 * Math.PI * variances[j][f]);
      const lw = Math.log(weights[j] + 1e-300);
      for (let i = 0; i < n; i += 1) {
        let quad = 0;
        for (let f = 0; f < d; f += 1) {
          const diff = X[i][f] - means[j][f];
          quad += (diff * diff) / variances[j][f];
        }
        logp[i][j] = lw - 0.5 * logDet - 0.5 * quad;
      }
    }

    const mass = new Float64Array(k);
    const resp = Array.from({ length: n }, () => new Float64Array(k));
    for (let i = 0; i < n; i += 1) {
      let top = -Infinity;
      for (let j = 0; j < k; j += 1) if (logp[i][j] > top) top = logp[i][j];
      let sum = 0;
      for (let j = 0; j < k; j += 1) { resp[i][j] = Math.exp(logp[i][j] - top); sum += resp[i][j]; }
      for (let j = 0; j < k; j += 1) { resp[i][j] /= sum; mass[j] += resp[i][j]; }
    }

    for (let j = 0; j < k; j += 1) {
      const m = mass[j] + 1e-300;
      weights[j] = mass[j] / n;
      const mean = new Array(d).fill(0);
      for (let i = 0; i < n; i += 1) for (let f = 0; f < d; f += 1) mean[f] += (resp[i][j] * X[i][f]) / m;
      means[j] = mean;
      const varr = new Array(d).fill(0);
      for (let i = 0; i < n; i += 1) {
        for (let f = 0; f < d; f += 1) {
          const diff = X[i][f] - mean[f];
          varr[f] += (resp[i][j] * diff * diff) / m;
        }
      }
      variances[j] = varr.map((v) => v + reg);
    }
  }

  const labels = new Int32Array(n);
  for (let i = 0; i < n; i += 1) {
    let best = 0, top = -Infinity;
    for (let j = 0; j < k; j += 1) if (logp[i][j] > top) { top = logp[i][j]; best = j; }
    labels[i] = best;
  }
  return labels;
}

/** Ward linkage by nearest-neighbour chain: one pass gives every k. */
export function wardTree(X) {
  const n = X.length, d = X[0].length;
  const centroids = new Map();
  const sizes = new Map();
  for (let i = 0; i < n; i += 1) { centroids.set(i, X[i].slice()); sizes.set(i, 1); }
  let active = Array.from({ length: n }, (_, i) => i);
  const merges = [];
  let nextId = n;

  const cost = (a, b) => {
    const ca = centroids.get(a), cb = centroids.get(b);
    let total = 0;
    for (let f = 0; f < d; f += 1) { const diff = ca[f] - cb[f]; total += diff * diff; }
    const sa = sizes.get(a), sb = sizes.get(b);
    return ((sa * sb) / (sa + sb)) * total;
  };

  let chain = [];
  while (active.length > 1) {
    if (!chain.length) chain = [active[0]];
    const a = chain[chain.length - 1];
    let best = null, bestCost = Infinity;
    for (const c of active) {
      if (c === a) continue;
      const dd = cost(a, c);
      if (dd < bestCost || (dd === bestCost && (best === null || c < best))) { best = c; bestCost = dd; }
    }
    if (chain.length >= 2 && chain[chain.length - 2] === best) {
      const b = chain.pop();
      const u = chain.pop();
      const merged = nextId; nextId += 1;
      const su = sizes.get(u), sb = sizes.get(b), total = su + sb;
      const cu = centroids.get(u), cb = centroids.get(b);
      const mean = new Array(d);
      for (let f = 0; f < d; f += 1) mean[f] = (cu[f] * su + cb[f] * sb) / total;
      centroids.set(merged, mean); sizes.set(merged, total);
      merges.push([u, b]);
      active = active.filter((x) => x !== u && x !== b);
      active.push(merged);
    } else {
      chain.push(best);
    }
    if (chain.length && !active.includes(chain[chain.length - 1])) chain = [];
  }
  return { merges, n };
}

export function wardLabels({ merges, n }, k) {
  const parent = new Map();
  merges.slice(0, n - k).forEach(([a, b], step) => { parent.set(a, n + step); parent.set(b, n + step); });
  const root = (i) => { while (parent.has(i)) i = parent.get(i); return i; };
  const seen = new Map();
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i += 1) {
    const r = root(i);
    if (!seen.has(r)) seen.set(r, seen.size);
    labels[i] = seen.get(r);
  }
  return labels;
}

// ---- the tests --------------------------------------------------------

function withinDispersion(X, labels) {
  const groups = new Map();
  for (let i = 0; i < X.length; i += 1) {
    if (!groups.has(labels[i])) groups.set(labels[i], []);
    groups.get(labels[i]).push(X[i]);
  }
  let total = 0;
  for (const members of groups.values()) {
    if (members.length < 2) continue;
    const d = members[0].length;
    const centre = new Array(d).fill(0);
    for (const m of members) for (let f = 0; f < d; f += 1) centre[f] += m[f] / members.length;
    for (const m of members) for (let f = 0; f < d; f += 1) total += (m[f] - centre[f]) ** 2;
  }
  return Math.log(total + 1e-300);
}

/** Tibshirani, Walther and Hastie (2001). */
export function gapStatistic(X, ks, seed, draws = 10, onStep) {
  const rng = mulberry32(seed);
  const n = X.length, d = X[0].length;
  const lo = new Array(d).fill(Infinity), hi = new Array(d).fill(-Infinity);
  for (const row of X) for (let f = 0; f < d; f += 1) {
    if (row[f] < lo[f]) lo[f] = row[f];
    if (row[f] > hi[f]) hi[f] = row[f];
  }

  const observed = ks.map((k) => withinDispersion(X, kmeans(X, k, seed).labels));
  const reference = ks.map(() => []);
  for (let b = 0; b < draws; b += 1) {
    const U = Array.from({ length: n }, () => {
      const row = new Array(d);
      for (let f = 0; f < d; f += 1) row[f] = lo[f] + (hi[f] - lo[f]) * rng();
      return row;
    });
    ks.forEach((k, i) => reference[i].push(withinDispersion(U, kmeans(U, k, seed + b).labels)));
    onStep?.();
  }

  const perK = ks.map((k, i) => {
    const ref = reference[i];
    const mean = ref.reduce((a, b) => a + b, 0) / ref.length;
    const sd = Math.sqrt(ref.reduce((a, b) => a + (b - mean) ** 2, 0) / ref.length);
    return { k, gap: mean - observed[i], s: sd * Math.sqrt(1 + 1 / draws),
             logW: observed[i], refLogW: mean };
  });

  // The standard rule: the smallest k whose gap is within one standard error
  // of the next k's. Taking the maximum over-selects.
  let chosenK = null;
  for (let i = 0; i < perK.length - 1; i += 1) {
    if (perK[i].gap >= perK[i + 1].gap - perK[i + 1].s) { chosenK = perK[i].k; break; }
  }
  return { perK, chosenK };
}

/** Agreement between two labellings, corrected for chance. */
export function adjustedRand(a, b) {
  const ua = [...new Set(a)], ub = [...new Set(b)];
  const ia = new Map(ua.map((v, i) => [v, i])), ib = new Map(ub.map((v, i) => [v, i]));
  const table = Array.from({ length: ua.length }, () => new Float64Array(ub.length));
  for (let i = 0; i < a.length; i += 1) table[ia.get(a[i])][ib.get(b[i])] += 1;

  const c2 = (x) => (x * (x - 1)) / 2;
  let sumIJ = 0, sumI = 0, sumJ = 0;
  for (let i = 0; i < ua.length; i += 1) {
    let rowTotal = 0;
    for (let j = 0; j < ub.length; j += 1) { sumIJ += c2(table[i][j]); rowTotal += table[i][j]; }
    sumI += c2(rowTotal);
  }
  for (let j = 0; j < ub.length; j += 1) {
    let colTotal = 0;
    for (let i = 0; i < ua.length; i += 1) colTotal += table[i][j];
    sumJ += c2(colTotal);
  }
  const total = c2(a.length);
  const expected = total ? (sumI * sumJ) / total : 0;
  const maximum = 0.5 * (sumI + sumJ);
  return maximum === expected ? 1 : (sumIJ - expected) / (maximum - expected);
}

const labeller = {
  kmeans: (X, k, seed) => kmeans(X, k, seed).labels,
  gmm: (X, k, seed) => gmmLabels(X, k, seed),
};

/** Bootstrap stability: does the same partition come back? */
export function stability(X, ks, seed, algorithm = "kmeans", draws = 6, onStep) {
  const rng = mulberry32(seed);
  const n = X.length;

  const base = new Map();
  if (algorithm === "ward") {
    const tree = wardTree(X);
    for (const k of ks) base.set(k, wardLabels(tree, k));
  } else {
    for (const k of ks) base.set(k, labeller[algorithm](X, k, seed));
  }

  const scores = new Map(ks.map((k) => [k, []]));
  for (let b = 0; b < draws; b += 1) {
    const idx = Array.from({ length: n }, () => integers(rng, n));
    const Xb = idx.map((i) => X[i]);
    const seen = [...new Set(idx)].sort((x, y) => x - y);
    const tree = algorithm === "ward" ? wardTree(Xb) : null;
    for (const k of ks) {
      const lb = algorithm === "ward" ? wardLabels(tree, k) : labeller[algorithm](Xb, k, seed + b);
      const first = new Map();
      idx.forEach((original, position) => { if (!first.has(original)) first.set(original, lb[position]); });
      scores.get(k).push(adjustedRand(seen.map((i) => first.get(i)), seen.map((i) => base.get(k)[i])));
    }
    onStep?.();
  }

  return ks.map((k) => {
    const s = scores.get(k);
    const mean = s.reduce((a, b) => a + b, 0) / s.length;
    return { k, ari: mean, ariSd: Math.sqrt(s.reduce((a, b) => a + (b - mean) ** 2, 0) / s.length) };
  });
}

/** Silhouette per k for each algorithm, so the answer is not k-means' alone. */
export function byAlgorithm(X, ks, seed, onStep) {
  const out = {};
  for (const name of ["kmeans", "gmm", "ward"]) {
    const tree = name === "ward" ? wardTree(X) : null;
    out[name] = ks.map((k) => {
      const labels = name === "ward" ? wardLabels(tree, k) : labeller[name](X, k, seed);
      onStep?.();
      return { k, silhouette: silhouette(X, labels, 400, seed) };
    });
  }
  return out;
}
