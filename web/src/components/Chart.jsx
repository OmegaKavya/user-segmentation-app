import React, { useState } from "react";
import "./Chart.css";

const W = 920;
const H = 300;
const PAD = { top: 18, right: 20, bottom: 40, left: 52 };

// Silhouette against the null band, per k.
//
// One axis, two things on it: the measured line, and a shaded band for the
// null's mean plus or minus one standard deviation. If the line sits inside
// the band there is nothing there, and that is readable at a glance without
// knowing what a silhouette is.
export default function Chart({ results }) {
  const [hover, setHover] = useState(null);
  if (!results?.length) return null;

  const ks = results.map((r) => r.k);
  const values = results.flatMap((r) => [
    r.silhouette, r.nullMean + r.nullSd, r.nullMean - r.nullSd,
  ]);
  const lo = Math.min(0, ...values);
  const hi = Math.max(...values) * 1.15;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i) => PAD.left + (ks.length < 2 ? plotW / 2 : (i / (ks.length - 1)) * plotW);
  const y = (v) => PAD.top + (1 - (v - lo) / (hi - lo || 1)) * plotH;

  const line = results.map((r, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(r.silhouette)}`).join("");
  const bandTop = results.map((r, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(r.nullMean + r.nullSd)}`).join("");
  const bandBottom = results.slice().reverse()
    .map((r, i) => `L${x(results.length - 1 - i)},${y(r.nullMean - r.nullSd)}`).join("");

  const ticks = 4;
  const gridlines = Array.from({ length: ticks + 1 }, (_, i) => lo + ((hi - lo) * i) / ticks);

  return (
    <figure className="chart">
      <figcaption>
        Silhouette at each k, against the shuffled null
      </figcaption>

      <svg viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label="Silhouette score per number of clusters, compared with a null band from independently shuffled columns."
        onMouseLeave={() => setHover(null)}>
        {gridlines.map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)} className="grid" />
            <text x={PAD.left - 9} y={y(v) + 4} className="tick" textAnchor="end">{v.toFixed(2)}</text>
          </g>
        ))}

        <path d={`${bandTop}${bandBottom}Z`} className="band" />
        <path d={line} className="line" />

        {results.map((r, i) => (
          <g key={r.k}>
            <circle cx={x(i)} cy={y(r.silhouette)} r={hover === i ? 7 : 5} className="dot" />
            <rect x={x(i) - plotW / (2 * ks.length)} y={PAD.top}
              width={plotW / ks.length} height={plotH}
              fill="transparent" onMouseEnter={() => setHover(i)} />
            <text x={x(i)} y={H - 16} className="tick" textAnchor="middle">{r.k}</text>
          </g>
        ))}

        <text x={PAD.left} y={H - 2} className="axis-label">clusters (k)</text>
      </svg>

      <div className="chart-key">
        <span><i className="swatch line-swatch" />measured</span>
        <span><i className="swatch band-swatch" />null, mean ±1 sd</span>
      </div>

      <p className="chart-readout" aria-live="polite">
        {hover !== null ? (
          <>
            <b>k = {results[hover].k}</b>: silhouette {results[hover].silhouette.toFixed(4)},
            null {results[hover].nullMean.toFixed(4)} ± {results[hover].nullSd.toFixed(4)},
            which is <b>{results[hover].z >= 0 ? "+" : ""}{results[hover].z.toFixed(1)}</b> sd
            from it. Cluster sizes {results[hover].sizes.join(", ")}.
          </>
        ) : (
          <>Hover any k to read it. A line inside the band means no structure at that k.</>
        )}
      </p>
    </figure>
  );
}
