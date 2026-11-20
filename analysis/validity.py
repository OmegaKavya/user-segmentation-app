"""Is there anything to cluster?

k-means always returns clusters. Given uniform noise it returns tidy,
well-separated-looking clusters, and an elbow plot drawn from inertia alone
will usually suggest a number. Neither fact says the clusters exist.

The test here is the one that answers the question: compare the clustering of
the real data against the clustering of a null in which every column keeps its
own distribution exactly, but any relationship between columns is destroyed by
shuffling each column independently. If the real data scores no better than
that null, there is no joint structure, and whatever partition k-means
returned is a partition of a featureless cloud.

This file is the reference implementation. web/src/cluster.js reimplements it
for the browser and is checked against these numbers.
"""
import numpy as np

SEED = 42


class Mulberry32:
    """A tiny PRNG, duplicated exactly in web/src/cluster.js.

    numpy's generator and JavaScript's have nothing in common, so matching
    them is impossible and any claim that the two implementations agree would
    rest on statistics rather than on identity. Sharing one trivially
    reproducible generator instead means both draw the same numbers and the
    agreement can be checked exactly.
    """

    MASK = 0xFFFFFFFF

    def __init__(self, seed):
        self.a = seed & self.MASK

    def next(self):
        """mulberry32.

        Everything is kept as an unsigned 32 bit pattern. JavaScript's |0 and
        >>> 0 only reinterpret the sign of that pattern, Math.imul is
        multiplication modulo 2**32, and >>> is a shift of the same pattern,
        so working unsigned throughout reproduces it exactly. Note the
        precedence in `t + Math.imul(...) ^ t`: the addition happens first.
        """
        self.a = (self.a + 0x6D2B79F5) & self.MASK
        a = self.a
        t = ((a ^ (a >> 15)) * (1 | a)) & self.MASK
        t = (((t + (((t ^ (t >> 7)) * (61 | t)) & self.MASK)) & self.MASK) ^ t) & self.MASK
        return ((t ^ (t >> 14)) & self.MASK) / 4294967296.0

    def integers(self, n):
        return int(self.next() * n)

    def permutation(self, values):
        """Fisher-Yates, drawing in the same order as the JavaScript."""
        out = list(values)
        for i in range(len(out) - 1, 0, -1):
            j = self.integers(i + 1)
            out[i], out[j] = out[j], out[i]
        return np.array(out)

    def choice_weighted(self, weights):
        target = self.next() * float(np.sum(weights))
        total = 0.0
        for i, w in enumerate(weights):
            total += float(w)
            if total >= target:
                return i
        return len(weights) - 1

    def sample_without_replacement(self, n, size):
        return self.permutation(np.arange(n))[:size]
NULL_REPEATS = 20
SILHOUETTE_SAMPLE = 400


def kmeans(X, k, seed=SEED, iters=60):
    """Lloyd's algorithm with k-means++ seeding, written out so the browser
    can match it step for step."""
    rng = Mulberry32(seed)
    n = len(X)

    centres = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        if d2.sum() <= 0:
            centres.append(X[rng.integers(n)])
            continue
        centres.append(X[rng.choice_weighted(d2)])
    centres = np.array(centres, dtype=float)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = X[labels == j]
            if len(members):
                centres[j] = members.mean(0)
    inertia = float(((X - centres[labels]) ** 2).sum())
    return labels, inertia


def silhouette(X, labels, sample=SILHOUETTE_SAMPLE, seed=SEED):
    """Mean silhouette, on a subsample when the data is large.

    Silhouette is quadratic in the number of points, and a few hundred points
    estimate the mean closely enough for a yes or no answer.
    """
    rng = Mulberry32(seed)
    n = len(X)
    idx = rng.sample_without_replacement(n, sample) if n > sample else np.arange(n)
    Xs, ls = X[idx], labels[idx]
    if len(np.unique(ls)) < 2:
        return 0.0

    d = np.sqrt(((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1))
    scores = []
    for i in range(len(Xs)):
        own = ls[i] == ls
        own[i] = False
        if own.sum() == 0:
            scores.append(0.0)
            continue
        a = d[i][own].mean()
        b = min(d[i][ls == other].mean() for other in np.unique(ls) if other != ls[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def shuffled(X, rng):
    """Each column permuted on its own: marginals kept, joint structure gone."""
    return np.column_stack([rng.permutation(col) for col in X.T])


def assess(X, ks=range(2, 9), repeats=NULL_REPEATS, seed=SEED):
    """Per k, and one honest verdict across all of them.

    Testing seven values of k means seven chances to clear a threshold, and
    correcting that with Bonferroni is impossible here: with R shuffles the
    smallest attainable p is 1/(R+1), so no dataset could ever pass.

    The max-statistic permutation test handles it by construction. Each
    shuffled copy is scored at every k and only its best score is kept, which
    builds the null distribution of "the best silhouette anyone could get by
    searching over k". Comparing the real best against that distribution
    corrects for the search without assuming the null has any shape, and
    needs no correction afterwards.
    """
    rng = Mulberry32(seed)
    ks = list(ks)

    rows = []
    for k in ks:
        labels, inertia = kmeans(X, k, seed=seed)
        rows.append({
            "k": int(k),
            "inertia": inertia,
            "silhouette": silhouette(X, labels, seed=seed),
            "sizes": [int((labels == j).sum()) for j in range(k)],
        })

    null_by_k = {k: [] for k in ks}
    null_best = []
    for r in range(repeats):
        Z = shuffled(X, rng)
        scores = []
        for k in ks:
            zl, _ = kmeans(Z, k, seed=seed + r)
            scores.append(silhouette(Z, zl, seed=seed))
            null_by_k[k].append(scores[-1])
        null_best.append(max(scores))

    for row, k in zip(rows, ks):
        nulls = null_by_k[k]
        mean, sd = float(np.mean(nulls)), float(np.std(nulls))
        row["nullMean"] = mean
        row["nullSd"] = sd
        row["z"] = float((row["silhouette"] - mean) / sd) if sd > 0 else 0.0

    best = max(rows, key=lambda r: r["silhouette"])
    beaten = sum(1 for v in null_best if v >= best["silhouette"])
    verdict = {
        "bestK": best["k"],
        "bestSilhouette": best["silhouette"],
        "nullBestMean": float(np.mean(null_best)),
        # The +1 correction: with `repeats` shuffles the smallest p is
        # 1/(repeats+1), which is what makes `repeats` a real choice.
        "p": (1 + beaten) / (1 + repeats),
        "repeats": repeats,
    }
    return {"perK": rows, "verdict": verdict}
