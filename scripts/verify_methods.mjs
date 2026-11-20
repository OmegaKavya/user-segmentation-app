// The three added methods, against analysis/methods.py.
//
// Both sides draw from the same mulberry32, so these are compared as numbers,
// not as distributions.
import fs from "node:fs";
import { gapStatistic, stability, byAlgorithm, adjustedRand } from "../web/src/methods.js";

const fx = JSON.parse(fs.readFileSync("/tmp/methods_fixture.json", "utf8"));
let worst = 0, compared = 0, failures = 0;

const check = (label, got, want, tol = 1e-9) => {
  const scale = Math.max(1, Math.abs(want));
  const delta = Math.abs(got - want) / scale;
  worst = Math.max(worst, delta); compared += 1;
  if (delta > tol) { console.log(`  ${label}: ${got} vs ${want}`); failures += 1; }
};

for (const [name, d] of Object.entries(fx)) {
  const ks = d.ks;

  const gap = gapStatistic(d.X, ks, 42, d.gapDraws);
  gap.perK.forEach((r, i) => {
    check(`${name} gap k=${r.k}`, r.gap, d.gap.perK[i].gap);
    check(`${name} gap s k=${r.k}`, r.s, d.gap.perK[i].s);
  });
  if (gap.chosenK !== d.gap.chosenK) { console.log(`  ${name} chosen k: ${gap.chosenK} vs ${d.gap.chosenK}`); failures += 1; }

  for (const algo of Object.keys(d.stability)) {
    const got = stability(d.X, ks, 42, algo, d.stabilityDraws);
    got.forEach((r, i) => check(`${name} ${algo} ARI k=${r.k}`, r.ari, d.stability[algo][i].ari));
  }

  const algo = byAlgorithm(d.X, ks, 42);
  for (const n2 of Object.keys(d.byAlgorithm)) {
    algo[n2].forEach((r, i) => check(`${name} ${n2} sil k=${r.k}`, r.silhouette, d.byAlgorithm[n2][i].silhouette));
  }
  console.log(`${name.padEnd(6)} gap, stability (3 algorithms) and silhouettes all match`);
}

check("ARI identity", adjustedRand([0, 0, 1, 1], [0, 0, 1, 1]), 1);
check("ARI inverted", adjustedRand([0, 0, 1, 1], [1, 1, 0, 0]), 1);

console.log(`\n${compared} numbers compared, worst relative difference ${worst.toExponential(2)}`);
process.exit(failures === 0 && worst < 1e-9 ? 0 : 1);
