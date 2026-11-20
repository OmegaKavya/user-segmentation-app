import React, { useEffect, useState } from "react";
import Hero from "./components/Hero.jsx";
import Lab from "./components/Lab.jsx";
import Finding from "./components/Finding.jsx";
import Signature from "./components/Signature.jsx";
import "./App.css";

export default function App() {
  const [datasets, setDatasets] = useState(null);

  useEffect(() => {
    // A separate chunk, so the page paints before the data arrives.
    let live = true;
    import("./datasets.json")
      .then((m) => { if (live) setDatasets(m.default ?? m); })
      .catch(() => {});
    return () => { live = false; };
  }, []);

  return (
    <>
      <Hero />
      <main className="shell">
        <Lab datasets={datasets} />
        <Finding />
      </main>
      <footer className="footer">
        <hr className="rule" />
        <Signature note="Built and measured end to end" />
        <p>
          <a href="https://github.com/OmegaKavya/user-segmentation-app">Source and method</a>
          <span aria-hidden="true"> · </span>
          <a href="https://aggarwalkavya.web.app">More work</a>
        </p>
      </footer>
    </>
  );
}
