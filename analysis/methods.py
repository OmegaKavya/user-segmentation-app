"""Clustering algorithms and the three validity tests, written out.

Written out rather than imported from scikit-learn because the browser has to
run the identical procedure, and "identical" has to be checkable.

The three tests are deliberately different in kind. Agreeing, they are strong
evidence. Disagreeing, they say which assumption is doing the work, which is
more useful than a single number either way.
"""
import numpy as np

from validity import Mulberry32, silhouette


# ---- algorithms -------------------------------------------------------

def kmeans_labels(X, k, seed, iters=60):
    from validity import kmeans
    return kmeans(X, k, seed=seed, iters=iters)[0]


def gmm_labels(X, k, seed, iters=40, reg=1e-6):
    """Expectation maximisation with diagonal covariances.

    Diagonal rather than full: full covariance needs a matrix inverse per
    component per iteration, which is a lot of linear algebra to reimplement
    exactly in two languages for a question that does not turn on it. The
    assumption is stated rather than hidden, and it is what makes this
    genuinely different from k-means: components may have different widths
    along each axis, so it can find elongated groups that k-means splits.
    """
    rng = Mulberry32(seed)
    n, d = X.shape
    idx = rng.permutation(np.arange(n))[:k]
    means = X[idx].astype(float).copy()
    variances = np.tile(X.var(axis=0) + reg, (k, 1))
    weights = np.full(k, 1.0 / k)

    for _ in range(iters):
        # log N(x | mu, diag(var)), per component
        logp = np.empty((n, k))
        for j in range(k):
            diff = X - means[j]
            logp[:, j] = (
                np.log(weights[j] + 1e-300)
                - 0.5 * np.sum(np.log(2 * np.pi * variances[j]))
                - 0.5 * np.sum(diff * diff / variances[j], axis=1)
            )
        top = logp.max(axis=1, keepdims=True)
        resp = np.exp(logp - top)
        resp /= resp.sum(axis=1, keepdims=True)

        mass = resp.sum(axis=0) + 1e-300
        weights = mass / n
        means = (resp.T @ X) / mass[:, None]
        for j in range(k):
            diff = X - means[j]
            variances[j] = (resp[:, j] @ (diff * diff)) / mass[j] + reg

    return logp.argmax(axis=1)


def ward_tree(X):
    """Ward linkage by nearest-neighbour chain.

    Returns the merge order, so one pass gives every k at once: cutting the
    tree at seven heights costs no more than cutting it at one.
    """
    n = len(X)
    active = list(range(n))
    centroids = {i: X[i].astype(float).copy() for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    merges = []
    next_id = n

    def cost(a, b):
        diff = centroids[a] - centroids[b]
        return (sizes[a] * sizes[b]) / (sizes[a] + sizes[b]) * float(diff @ diff)

    chain = []
    while len(active) > 1:
        if not chain:
            chain = [active[0]]
        a = chain[-1]
        best, best_cost = None, np.inf
        for c in active:
            if c == a:
                continue
            d = cost(a, c)
            if d < best_cost or (d == best_cost and (best is None or c < best)):
                best, best_cost = c, d
        if len(chain) >= 2 and chain[-2] == best:
            b = chain.pop()
            a = chain.pop()
            merged = next_id
            next_id += 1
            total = sizes[a] + sizes[b]
            centroids[merged] = (centroids[a] * sizes[a] + centroids[b] * sizes[b]) / total
            sizes[merged] = total
            merges.append((a, b))
            active.remove(a)
            active.remove(b)
            active.append(merged)
        else:
            chain.append(best)
        if chain and chain[-1] not in active:
            chain = []
    return merges, n


def ward_labels_from_tree(merges, n, k):
    """Cut the tree so that k clusters remain."""
    parent = {}
    for step, (a, b) in enumerate(merges[: n - k]):
        parent[a] = n + step
        parent[b] = n + step

    def root(i):
        while i in parent:
            i = parent[i]
        return i

    roots = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        r = root(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels


ALGORITHMS = {
    "kmeans": kmeans_labels,
    "gmm": gmm_labels,
    "ward": None,  # handled separately: one tree gives every k
}


# ---- the three tests --------------------------------------------------

def within_dispersion(X, labels):
    """log of the pooled within-cluster sum of squares, the quantity the gap
    statistic compares."""
    total = 0.0
    for j in np.unique(labels):
        members = X[labels == j]
        if len(members) < 2:
            continue
        centre = members.mean(axis=0)
        total += float(((members - centre) ** 2).sum())
    return np.log(total + 1e-300)


def gap_statistic(X, ks, seed, draws=20):
    """Tibshirani, Walther and Hastie (2001).

    Compares log within-cluster dispersion against the same quantity on data
    drawn uniformly from the bounding box of the real data. The reference has
    no clusters by construction but keeps the data's extent, so a gap that
    does not open means the compactness is what spread alone would give.

    The standard rule picks the smallest k whose gap is within one standard
    error of the next k's, rather than the maximum, which over-selects.
    """
    rng = Mulberry32(seed)
    lo, hi = X.min(axis=0), X.max(axis=0)
    n, d = X.shape

    observed = {k: within_dispersion(X, kmeans_labels(X, k, seed)) for k in ks}

    reference = {k: [] for k in ks}
    for b in range(draws):
        # Row by row, feature by feature. The order matters: the JavaScript
        # port fills the same array in the same order, and drawing column
        # first would consume the shared random stream differently and make
        # the two disagree for no reason anyone could see.
        U = np.empty((n, d))
        for i in range(n):
            for f in range(d):
                U[i, f] = lo[f] + (hi[f] - lo[f]) * rng.next()
        for k in ks:
            reference[k].append(within_dispersion(U, kmeans_labels(U, k, seed + b)))

    out = []
    for k in ks:
        ref = np.array(reference[k])
        gap = float(ref.mean() - observed[k])
        sd = float(ref.std())
        out.append({"k": int(k), "gap": gap,
                    "s": float(sd * np.sqrt(1 + 1 / draws)),
                    "logW": float(observed[k]),
                    "refLogW": float(ref.mean())})

    chosen = None
    for i in range(len(out) - 1):
        if out[i]["gap"] >= out[i + 1]["gap"] - out[i + 1]["s"]:
            chosen = out[i]["k"]
            break
    return {"perK": out, "chosenK": chosen}


def adjusted_rand(a, b):
    """Agreement between two labellings, corrected for chance."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    table = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            table[i, j] = np.sum((a == x) & (b == y))

    def comb2(x):
        return x * (x - 1) / 2

    sum_ij = comb2(table).sum()
    sum_i = comb2(table.sum(axis=1)).sum()
    sum_j = comb2(table.sum(axis=0)).sum()
    total = comb2(len(a))
    expected = sum_i * sum_j / total if total else 0.0
    maximum = 0.5 * (sum_i + sum_j)
    return float((sum_ij - expected) / (maximum - expected)) if maximum != expected else 1.0


def stability(X, ks, seed, algorithm="kmeans", draws=10):
    """Bootstrap stability.

    Resample with replacement, cluster the resample, and ask how far the two
    labellings agree on the points they share. Real groups survive having
    some of their members removed and others duplicated; a partition of noise
    is redrawn somewhere else each time.

    This assumes nothing about a null distribution, which is what makes it
    independent evidence rather than a second permutation test.
    """
    rng = Mulberry32(seed)
    n = len(X)

    base = {}
    if algorithm == "ward":
        merges, _ = ward_tree(X)
        for k in ks:
            base[k] = ward_labels_from_tree(merges, n, k)
    else:
        fn = ALGORITHMS[algorithm]
        for k in ks:
            base[k] = fn(X, k, seed)

    scores = {k: [] for k in ks}
    for b in range(draws):
        idx = np.array([rng.integers(n) for _ in range(n)])
        seen = np.unique(idx)
        Xb = X[idx]
        if algorithm == "ward":
            merges_b, _ = ward_tree(Xb)
        for k in ks:
            if algorithm == "ward":
                lb = ward_labels_from_tree(merges_b, len(Xb), k)
            else:
                lb = ALGORITHMS[algorithm](Xb, k, seed + b)
            # One label per original point: the first time it appears.
            first = {}
            for position, original in enumerate(idx):
                first.setdefault(original, lb[position])
            mine = np.array([first[i] for i in seen])
            theirs = base[k][seen]
            scores[k].append(adjusted_rand(mine, theirs))

    return [{"k": int(k), "ari": float(np.mean(scores[k])),
             "ariSd": float(np.std(scores[k]))} for k in ks]


def by_algorithm(X, ks, seed):
    """Silhouette per k for each algorithm, so the answer is not k-means'
    alone. A dataset with genuine non-convex structure can defeat k-means and
    still be found by Ward."""
    out = {}
    n = len(X)
    for name in ("kmeans", "gmm", "ward"):
        rows = []
        if name == "ward":
            merges, _ = ward_tree(X)
        for k in ks:
            labels = (ward_labels_from_tree(merges, n, k) if name == "ward"
                      else ALGORITHMS[name](X, k, seed))
            rows.append({"k": int(k), "silhouette": silhouette(X, labels, seed=seed)})
        out[name] = rows
    return out
