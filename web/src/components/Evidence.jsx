import React from "react";
import "./Evidence.css";

const fmt = (v, n = 3) => (v === null || v === undefined ? "n/a" : v.toFixed(n));

// Conventional reading of the adjusted Rand index. Stated as a convention
// rather than a test, because it is one.
function stabilityWord(ari) {
  if (ari >= 0.75) return ["reproducible", "yes"];
  if (ari >= 0.6) return ["partly reproducible", "mid"];
  return ["not reproducible", "no"];
}

export function Gap({ gap }) {
  if (!gap) return <Pending title="Gap statistic" />;
  const best = gap.perK.reduce((a, b) => (b.gap > a.gap ? b : a));
  const chose = gap.chosenK;
  const degenerate = chose === gap.perK[0].k;
  return (
    <article className={`ev ${degenerate ? "no" : "yes"}`}>
      <h3>Gap statistic</h3>
      <p className="ev-what">
        Compares how tight the clusters are against the same measure on data
        drawn uniformly across the same range. Tibshirani, Walther and Hastie,
        2001.
      </p>
      <p className="ev-verdict">
        {degenerate
          ? `Selects k = ${chose}, the smallest it is offered, which is what the rule does when no gap ever opens.`
          : `Selects k = ${chose}. The gap peaks at ${fmt(best.gap, 2)} at k = ${best.k}.`}
      </p>
      <ul className="ev-rows">
        {gap.perK.map((r) => (
          <li key={r.k} className={r.k === chose ? "on" : ""}>
            <span>k = {r.k}</span>
            <span className="num">{fmt(r.gap, 3)}</span>
            <span className="num dim">± {fmt(r.s, 3)}</span>
          </li>
        ))}
      </ul>
    </article>
  );
}

export function Stability({ stability: s }) {
  if (!s) return <Pending title="Bootstrap stability" />;
  const names = { kmeans: "k-means", gmm: "Gaussian mixture", ward: "Ward linkage" };
  const best = Object.entries(s).map(([algo, rows]) => {
    const top = rows.reduce((a, b) => (b.ari > a.ari ? b : a));
    return { algo, ...top };
  });
  const peak = best.reduce((a, b) => (b.ari > a.ari ? b : a));
  const [word, tone] = stabilityWord(peak.ari);

  return (
    <article className={`ev ${tone}`}>
      <h3>Bootstrap stability</h3>
      <p className="ev-what">
        Resamples the data, clusters it again, and measures how far the two
        labellings agree on shared points. Real groups survive resampling;
        a partition of noise is redrawn somewhere else.
      </p>
      <p className="ev-verdict">
        Best agreement is <b>{fmt(peak.ari, 2)}</b> at k = {peak.k} under{" "}
        {names[peak.algo]}, which is <b>{word}</b>. Above 0.75 is conventionally
        read as stable, below 0.6 as unstable.
      </p>
      <ul className="ev-rows">
        {best.map((r) => (
          <li key={r.algo}>
            <span>{names[r.algo]}</span>
            <span className="num">{fmt(r.ari, 3)}</span>
            <span className="num dim">k = {r.k}</span>
          </li>
        ))}
      </ul>
    </article>
  );
}

export function Algorithms({ algorithms }) {
  if (!algorithms) return <Pending title="Across three algorithms" />;
  const names = { kmeans: "k-means", gmm: "Gaussian mixture", ward: "Ward linkage" };
  const peaks = Object.entries(algorithms).map(([algo, rows]) => {
    const top = rows.reduce((a, b) => (b.silhouette > a.silhouette ? b : a));
    return { algo, ...top };
  });
  const agree = new Set(peaks.map((p) => p.k)).size === 1;
  const strong = peaks.some((p) => p.silhouette > 0.25);

  return (
    <article className={`ev ${strong ? "yes" : "no"}`}>
      <h3>Across three algorithms</h3>
      <p className="ev-what">
        k-means assumes round, equal-sized groups. A Gaussian mixture allows
        different widths per axis, and Ward linkage builds groups by merging
        rather than by distance to a centre. If only one of them finds
        something, that something is probably its assumption.
      </p>
      <p className="ev-verdict">
        {agree
          ? `All three peak at k = ${peaks[0].k}.`
          : `They peak at different k: ${peaks.map((p) => `${names[p.algo]} at ${p.k}`).join(", ")}.`}{" "}
        {strong
          ? "The silhouettes are high enough to mean something."
          : "Every silhouette is near zero, which is what a featureless cloud gives whatever is run on it."}
      </p>
      <ul className="ev-rows">
        {peaks.map((p) => (
          <li key={p.algo}>
            <span>{names[p.algo]}</span>
            <span className="num">{fmt(p.silhouette, 3)}</span>
            <span className="num dim">k = {p.k}</span>
          </li>
        ))}
      </ul>
    </article>
  );
}

function Pending({ title }) {
  return (
    <article className="ev pending">
      <h3>{title}</h3>
      <p className="ev-what">Running…</p>
    </article>
  );
}

/** What the four tests say taken together.
 *
 * Counted rather than argued: each test contributes one vote, and the summary
 * says how many agreed. Four tests agreeing is a much stronger claim than any
 * one of them, and four disagreeing is the interesting case, because it says
 * which assumption is carrying the result.
 */
export function Synthesis({ permutation, gap, stability: s, algorithms }) {
  if (!permutation || !gap || !s || !algorithms) return null;

  const votes = [];
  votes.push({
    name: "Permutation test",
    yes: permutation.verdict.p <= 0.05,
    detail: `p = ${permutation.verdict.p.toFixed(3)}`,
  });
  votes.push({
    name: "Gap statistic",
    yes: gap.chosenK !== gap.perK[0].k,
    detail: `selects k = ${gap.chosenK}`,
  });
  const peakAri = Object.values(s).flat().reduce((a, b) => (b.ari > a.ari ? b : a));
  votes.push({
    name: "Bootstrap stability",
    yes: peakAri.ari >= 0.75,
    detail: `best ARI ${peakAri.ari.toFixed(2)}`,
  });
  const peaks = Object.values(algorithms).map((rows) =>
    rows.reduce((a, b) => (b.silhouette > a.silhouette ? b : a)));
  const agreeK = new Set(peaks.map((p) => p.k)).size === 1;
  votes.push({
    name: "Three algorithms",
    yes: agreeK && peaks.some((p) => p.silhouette > 0.25),
    detail: agreeK ? `all peak at k = ${peaks[0].k}` : "peak at different k",
  });

  const yes = votes.filter((v) => v.yes).length;
  const unanimous = yes === 0 || yes === votes.length;

  return (
    <div className={`synth ${yes >= 3 ? "yes" : "no"}`}>
      <p className="eyebrow">Taken together</p>
      <p className="synth-line">
        {unanimous
          ? `All four tests agree: ${yes === 0 ? "there is no structure here." : "the groups are real."}`
          : `${yes} of ${votes.length} tests find structure, which is the interesting case: the disagreement says the result depends on an assumption rather than on the data.`}
      </p>
      <ul className="synth-votes">
        {votes.map((v) => (
          <li key={v.name} className={v.yes ? "yes" : "no"}>
            <i aria-hidden="true">{v.yes ? "✓" : "✗"}</i>
            <b>{v.name}</b>
            <span>{v.detail}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
