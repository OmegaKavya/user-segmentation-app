import React from "react";
import ThemeToggle from "./ThemeToggle.jsx";
import "./Hero.css";

export default function Hero() {
  return (
    <header className="hero">
      <ThemeToggle />
      <div className="hero-inner shell">
        <p className="hero-kicker">k-means always returns clusters</p>
        <h1>Are these clusters real?</h1>
        <p className="hero-tag">
          Give it uniform noise and it will hand back four tidy groups, and an
          elbow plot will usually suggest a number. Neither says the groups
          exist. This runs the test that does, in your browser.
        </p>

        <dl className="hero-stats">
          <div>
            <dt className="no">0.043</dt>
            <dd>silhouette of the four segments this project shipped</dd>
          </div>
          <div>
            <dt className="no">0.042</dt>
            <dd>what the same data scores once every column is shuffled, breaking every relationship in it</dd>
          </div>
          <div>
            <dt className="no">p = 0.19</dt>
            <dd>the chance of seeing that much structure in data with none</dd>
          </div>
        </dl>
      </div>
    </header>
  );
}
