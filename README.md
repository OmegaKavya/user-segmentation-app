# Are These Clusters Real?

**[Live](https://are-these-clusters-real.web.app)** · a cluster validity test
that runs in the browser.

This began as a user segmentation dashboard. It clustered a thousand
ad-audience profiles into four groups, named them Power Users, Premium
Engagers, Digital Natives and Casual Browsers, and reported on their
behaviour. The groups do not exist.

## What was wrong

**The names were assigned by cluster index.** k-means numbers its clusters
arbitrarily, so `0: "Digital Natives"` attached a behavioural claim to
whichever cluster happened to come out first. Nothing checked the names
against what the clusters did.

**Checked, the four segments are indistinguishable:**

| Segment | hrs/day | CTR | conversion | likes |
| --- | ---: | ---: | ---: | ---: |
| Power Users | 2.81 | 0.129 | 0.048 | 4,949 |
| Premium Engagers | 2.73 | 0.131 | 0.053 | 4,829 |
| Digital Natives | 2.75 | 0.126 | 0.050 | 4,986 |
| Casual Browsers | 2.75 | 0.122 | 0.049 | 5,071 |

The Power Users spend three minutes a day longer online than the Casual
Browsers. The Premium Engagers have the fewest likes.

**The elbow curve was computed and ignored.** The training script evaluates
inertia for k from 1 to 10 and then sets `optimal_k = 4` without consulting
it. Inertia falls smoothly the whole way; there is no elbow.

**The data cannot support segments.** Every categorical field is close to
uniform, the largest correlation between any two behavioural columns is 0.08,
and click-through rate does not differ by age, gender, income or education
(all p > 0.38). The columns were generated independently, so there is no joint
structure to find.

## The test

k-means always returns clusters. Given uniform noise it returns tidy ones.
The question is whether they are more than a partition of a featureless cloud,
and the way to answer it is to compare against a null that has no structure by
construction.

Each column is shuffled independently. Every column keeps its own
distribution exactly; only the relationships between columns are destroyed,
which is precisely what clustering claims to find. If the real data scores no
better than that, there is nothing there.

Searching over seven values of k is seven chances to clear a threshold, so the
comparison uses a **max-statistic permutation test**: each shuffled copy is
scored at every k and only its best is kept, building the null distribution of
"the best silhouette obtainable by searching over k". No correction is needed
afterwards, and none would be possible, since with R shuffles the smallest
attainable p is 1/(R+1).

| | best k | silhouette | best null | p |
| --- | ---: | ---: | ---: | ---: |
| Ad audience, 1000 users | 5 | 0.058 | 0.051 | **0.19** |
| Customer tiers, 900 records (control) | 3 | 0.446 | 0.108 | **0.032** |

The control exists because a test that only ever says "no" is
indistinguishable from a broken test. Both run in the page, on the same code.

## No server

The whole assessment is arithmetic, so it runs client side in a Web Worker:
30 shuffles at each of 7 values of k, around 6 seconds, with nothing leaving
the browser. You can upload your own CSV and it is never transmitted anywhere.

`analysis/validity.py` is the reference implementation and `web/src/cluster.js`
is the port. They share one small deterministic generator, so they draw the
same random numbers and agree exactly rather than approximately:
`scripts/verify_cluster.mjs` compares 38 numbers and the worst relative
difference is 5e-16.

## Running it

```bash
pip install -r requirements.txt

python scripts/bake_datasets.py      # prepare the bundled datasets
node scripts/verify_cluster.mjs      # JS against the Python reference

cd web && npm install && npm run dev
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the method and the decisions behind
it.
