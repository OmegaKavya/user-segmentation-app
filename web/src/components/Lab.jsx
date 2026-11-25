import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { parseCsv, numericColumns, matrix } from "../csv.js";
import Panel from "./Panel.jsx";
import Chart from "./Chart.jsx";
import Verdict from "./Verdict.jsx";
import { Gap, Stability, Algorithms, Synthesis } from "./Evidence.jsx";
import "./Lab.css";

const KS = [2, 3, 4, 5, 6, 7, 8];
const REPEATS = 30;

const STAGE_LABEL = {
  permutation: "the permutation test",
  gap: "the gap statistic",
  algorithms: "three clustering algorithms",
  stability: "bootstrap stability",
};

export default function Lab({ datasets }) {
  const [key, setKey] = useState("ads");
  // The parsed upload and which of its numeric columns are in play. The
  // dataset actually tested is derived from these, so ticking a column
  // re-runs the tests on the new selection.
  const [upload, setUpload] = useState(null);
  const [state, setState] = useState({ status: "idle" });
  const workerRef = useRef(null);

  const custom = useMemo(() => {
    if (!upload) return null;
    const chosen = upload.columns.filter((c) => c.include);
    if (chosen.length < 2) return { error: "Choose at least two columns." };
    const rows = matrix(upload.rows, chosen);
    if (rows.length < 30) return { error: "Needs at least 30 rows complete in the chosen columns." };
    return {
      title: upload.name,
      note: `${rows.length} rows, ${chosen.length} numeric columns, standardised.`,
      columns: chosen.map((c) => c.name),
      X: standardise(rows),
    };
  }, [upload]);

  const usable = custom && !custom.error ? custom : null;
  const active = key === "custom" ? usable : datasets?.[key];

  const run = useCallback((data) => {
    if (!data) return;
    workerRef.current?.terminate();
    const worker = new Worker(new URL("../worker.js", import.meta.url), { type: "module" });
    workerRef.current = worker;
    setState({ status: "running", stage: "permutation" });
    // Each test reports as it finishes, so the page fills in rather than
    // waiting on the slowest one.
    worker.onmessage = (event) => {
      const m = event.data;
      setState((s) => {
        switch (m.type) {
          case "stage": return { ...s, status: "running", stage: m.stage, sampled: m.sampled ?? s.sampled };
          case "permutation": return { ...s, status: "running", permutation: m.result };
          case "gap": return { ...s, gap: m.result };
          case "algorithms": return { ...s, algorithms: m.result };
          case "stability": return { ...s, stability: m.result };
          case "done": return { ...s, status: "done" };
          case "error": return { status: "failed", message: m.message };
          default: return s;
        }
      });
    };
    worker.postMessage({ X: data.X, ks: KS, repeats: REPEATS });
  }, []);

  useEffect(() => { if (active) run(active); }, [active, run]);
  useEffect(() => () => workerRef.current?.terminate(), []);

  const onFile = async (file) => {
    if (!file) return;
    try {
      const rows = parseCsv(await file.text());
      if (rows.length < 2) throw new Error("That file has no data rows.");
      // Numeric columns only. Anything else would need an encoding decision,
      // and guessing one silently is how this project went wrong the first time.
      const columns = numericColumns(rows);
      if (columns.length < 2) throw new Error("Needs at least two numeric columns.");
      setUpload({ name: file.name, rows, columns });
      setKey("custom");
    } catch (error) {
      setState({ status: "failed", message: error.message });
    }
  };

  const toggleColumn = (index) => setUpload((u) => ({
    ...u,
    columns: u.columns.map((c) => (c.index === index ? { ...c, include: !c.include } : c)),
  }));

  return (
    <Panel className="lab" aria-labelledby="lab-heading">
      <h2 id="lab-heading">Test a dataset</h2>
      <p className="note">
        k-means is run for every k from {KS[0]} to {KS[KS.length - 1]}, then again
        on {REPEATS} copies of the data with each column shuffled independently. The shuffle keeps
        every column's own distribution and destroys only the relationships
        between them, which is exactly the thing clustering claims to find.
      </p>

      <div className="lab-picks">
        {datasets && Object.entries(datasets).map(([id, d]) => (
          <button key={id} type="button" className={`pick${key === id ? " on" : ""}`}
            onClick={() => setKey(id)}>{d.title}</button>
        ))}
        {upload && (
          <button type="button" className={`pick${key === "custom" ? " on" : ""}`}
            onClick={() => setKey("custom")}>{upload.name}</button>
        )}
        <label className="pick upload">
          Upload a CSV
          <input type="file" accept=".csv,text/csv"
            onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
        </label>
      </div>

      {key === "custom" && upload && (
        <fieldset className="lab-columns">
          <legend>Columns to cluster on</legend>
          {upload.columns.map((c) => (
            <label key={c.index} className="lab-column">
              <input type="checkbox" checked={c.include} onChange={() => toggleColumn(c.index)} />
              {c.name}
            </label>
          ))}
          <p className="lab-columns-note">
            Columns named like ids, postcodes or phone numbers start unticked:
            they are labels, and clustering on them finds how the file was numbered.
          </p>
        </fieldset>
      )}
      {key === "custom" && custom?.error && <p className="lab-error" role="alert">{custom.error}</p>}

      {active && <p className="lab-about">{active.note}</p>}

      {state.status === "running" && (
        <p className="lab-running" role="status">
          Running {STAGE_LABEL[state.stage] || "the tests"}…
        </p>
      )}

      {state.status === "failed" && <p className="lab-error" role="alert">{state.message}</p>}

      {state.permutation && (
        <>
          <Verdict result={state.permutation} />
          <Chart results={state.permutation.perK} />

          <h3 className="lab-sub">Two more lines of evidence</h3>
          <p className="lab-sub-note">
            The test above compares against a null with no structure. These ask
            different questions, so agreement between them means more than any
            one of them alone.
            {state.sampled && ` Computed on a ${state.sampled} point sample, which is ample for the answer and keeps the run to seconds.`}
          </p>
          <div className="evidence-grid">
            <Gap gap={state.gap} />
            <Stability stability={state.stability} />
            <Algorithms algorithms={state.algorithms} />
          </div>
          <Synthesis
            permutation={state.permutation}
            gap={state.gap}
            stability={state.stability}
            algorithms={state.algorithms}
          />
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
