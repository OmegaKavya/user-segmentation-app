import React, { useCallback, useEffect, useRef, useState } from "react";
import Panel from "./Panel.jsx";
import Chart from "./Chart.jsx";
import Verdict from "./Verdict.jsx";
import "./Lab.css";

const KS = [2, 3, 4, 5, 6, 7, 8];
const REPEATS = 30;

export default function Lab({ datasets }) {
  const [key, setKey] = useState("ads");
  const [custom, setCustom] = useState(null);
  const [state, setState] = useState({ status: "idle" });
  const workerRef = useRef(null);

  const active = custom && key === "custom" ? custom : datasets?.[key];

  const run = useCallback((data) => {
    if (!data) return;
    workerRef.current?.terminate();
    const worker = new Worker(new URL("../worker.js", import.meta.url), { type: "module" });
    workerRef.current = worker;
    setState({ status: "running", progress: 0 });
    worker.onmessage = (event) => {
      const m = event.data;
      if (m.type === "progress") setState((s) => ({ ...s, progress: m.fraction }));
      else if (m.type === "done") setState({ status: "done", results: m.results });
      else setState({ status: "failed", message: m.message });
    };
    worker.postMessage({ X: data.X, ks: KS, repeats: REPEATS });
  }, []);

  useEffect(() => { if (active) run(active); }, [active, run]);
  useEffect(() => () => workerRef.current?.terminate(), []);

  const onFile = async (file) => {
    if (!file) return;
    try {
      const text = await file.text();
      const rows = text.trim().split(/\r?\n/).map((r) => r.split(","));
      const header = rows[0].map((h) => h.trim());
      // Numeric columns only. Anything else would need an encoding decision,
      // and guessing one silently is how this project went wrong the first time.
      // Numeric columns are taken by index, so a column whose name repeats
      // cannot be picked twice or missed.
      const numeric = header
        .map((_, c) => c)
        .filter((c) => rows.slice(1, 40).every((r) => {
          const cell = r[c];
          return cell !== undefined && cell.trim() !== "" && !Number.isNaN(Number(cell));
        }));
      if (numeric.length < 2) throw new Error("Needs at least two numeric columns.");

      const body = rows.slice(1).filter((r) => numeric.every((c) => r[c] !== undefined && r[c].trim() !== ""));
      if (body.length < 30) throw new Error("Needs at least 30 complete rows.");

      const columns = numeric.map((c) => header[c]);
      const X = standardise(body.map((r) => numeric.map((c) => Number(r[c]))));
      setCustom({ title: file.name, note: `${X.length} rows, ${columns.length} numeric columns, standardised.`, columns, X });
      setKey("custom");
    } catch (error) {
      setState({ status: "failed", message: error.message });
    }
  };

  return (
    <Panel className="lab" aria-labelledby="lab-heading">
      <h2 id="lab-heading">Test a dataset</h2>
      <p className="note">
        k-means is run for every k from 2 to 8, then again on twenty copies of
        the data with each column shuffled independently. The shuffle keeps
        every column's own distribution and destroys only the relationships
        between them, which is exactly the thing clustering claims to find.
      </p>

      <div className="lab-picks">
        {datasets && Object.entries(datasets).map(([id, d]) => (
          <button key={id} type="button" className={`pick${key === id ? " on" : ""}`}
            onClick={() => setKey(id)}>{d.title}</button>
        ))}
        {custom && (
          <button type="button" className={`pick${key === "custom" ? " on" : ""}`}
            onClick={() => setKey("custom")}>{custom.title}</button>
        )}
        <label className="pick upload">
          Upload a CSV
          <input type="file" accept=".csv,text/csv"
            onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
        </label>
      </div>

      {active && <p className="lab-about">{active.note}</p>}

      {state.status === "running" && (
        <div className="lab-progress" role="status">
          <div className="bar"><span style={{ width: `${Math.round(100 * (state.progress || 0))}%` }} /></div>
          <p>Clustering {KS.length} values of k, plus {KS.length * REPEATS} shuffled nulls…</p>
        </div>
      )}

      {state.status === "failed" && <p className="lab-error" role="alert">{state.message}</p>}

      {state.status === "done" && (
        <>
          <Verdict result={state.results} />
          <Chart results={state.results.perK} />
        </>
      )}
    </Panel>
  );
}

/** Centre and scale each column, which is what makes distances comparable. */
function standardise(rows) {
  const n = rows.length, d = rows[0].length;
  const mean = new Array(d).fill(0);
  for (const r of rows) for (let c = 0; c < d; c += 1) mean[c] += r[c] / n;
  const sd = new Array(d).fill(0);
  for (const r of rows) for (let c = 0; c < d; c += 1) sd[c] += (r[c] - mean[c]) ** 2 / n;
  for (let c = 0; c < d; c += 1) sd[c] = Math.sqrt(sd[c]) || 1;
  return rows.map((r) => r.map((v, c) => (v - mean[c]) / sd[c]));
}
