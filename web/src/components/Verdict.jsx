import React from "react";
import "./Verdict.css";

// The headline. A silhouette that does not beat the shuffled null means the
// partition is arbitrary, however tidy the chart looks, so that comparison is
// what gets the large type rather than the silhouette on its own.
// Across seven values of k, so the threshold is tightened for the seven
// chances taken: 0.05/7 by Bonferroni, which 1/21 cannot reach. In practice a
// result only passes when it is far clear of the null, which is the correct
// bar for a claim that groups exist.
// The permutation test already accounts for searching over k, so this is a
// plain threshold rather than a corrected one.
const ALPHA = 0.05;

export function readVerdict(result) {
  if (!result?.verdict) return null;
  const { verdict, perK } = result;
  const peak = perK.find((r) => r.k === verdict.bestK) ?? perK[0];
  return { verdict, peak, real: verdict.p <= ALPHA };
}

export default function Verdict({ result }) {
  const v = readVerdict(result);
  if (!v) return null;

  return (
    <div className={`verdict ${v.real ? "yes" : "no"}`}>
      <p className="eyebrow">The answer</p>
      <h3>{v.real ? "Yes, there is structure here" : "No, these clusters are not real"}</h3>

      {v.real ? (
        <p>
          At <b>k = {v.verdict.bestK}</b> the silhouette is{" "}
          <b>{v.verdict.bestSilhouette.toFixed(3)}</b>, against{" "}
          {v.verdict.nullBestMean.toFixed(3)} for the best a shuffled copy
          managed at any k. <b>p = {v.verdict.p.toFixed(3)}</b> over{" "}
          {v.verdict.repeats} shuffles. The structure is in how the columns
          relate to each other, not in any one of them.
        </p>
      ) : (
        <p>
          The best any k manages is a silhouette of{" "}
          <b>{v.verdict.bestSilhouette.toFixed(3)}</b> at k = {v.verdict.bestK},
          against {v.verdict.nullBestMean.toFixed(3)} for the best a shuffled
          copy reached. <b>p = {v.verdict.p.toFixed(3)}</b> over{" "}
          {v.verdict.repeats} shuffles, which is no evidence at all. Destroy
          every relationship between the columns and k-means scores the same,
          so the partition it returns is a partition of a featureless cloud.
        </p>
      )}
    </div>
  );
}
