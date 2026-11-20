// Does the browser implementation agree with the Python reference?
//
// Both draw from the same generator, so this is not a statistical comparison:
// the numbers must match to floating point.
import fs from "node:fs";
import { assess } from "../web/src/cluster.js";

const fixture = JSON.parse(fs.readFileSync("/tmp/cluster_fixture.json", "utf8"));
let worst = 0;
let rows = 0;

for (const [name, { X, expected }] of Object.entries(fixture)) {
  const got = assess(X, { ks: expected.perK.map((e) => e.k), repeats: expected.verdict.repeats });
  for (let i = 0; i < expected.perK.length; i += 1) {
    for (const field of ["silhouette", "nullMean", "nullSd", "inertia"]) {
      const scale = field === "inertia" ? Math.abs(expected.perK[i][field]) || 1 : 1;
      worst = Math.max(worst, Math.abs(got.perK[i][field] - expected.perK[i][field]) / scale);
      rows += 1;
    }
    const a = JSON.stringify(got.perK[i].sizes), b = JSON.stringify(expected.perK[i].sizes);
    if (a !== b) { console.log(`cluster sizes differ for ${name} k=${expected.perK[i].k}: ${a} vs ${b}`); process.exit(1); }
  }
  for (const field of ["bestSilhouette", "nullBestMean", "p"]) {
    worst = Math.max(worst, Math.abs(got.verdict[field] - expected.verdict[field]));
    rows += 1;
  }
  if (got.verdict.bestK !== expected.verdict.bestK) {
    console.log(`best k differs for ${name}: ${got.verdict.bestK} vs ${expected.verdict.bestK}`); process.exit(1);
  }
  console.log(`${name.padEnd(8)} best k=${got.verdict.bestK}, p=${got.verdict.p.toFixed(4)}, cluster sizes identical`);
}
console.log(`\n${rows} numbers compared, worst relative difference ${worst.toExponential(2)}`);
process.exit(worst < 1e-9 ? 0 : 1);
