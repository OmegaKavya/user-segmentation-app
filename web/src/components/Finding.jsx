import React from "react";
import Panel from "./Panel.jsx";
import "./Finding.css";

// Measured on the dataset this project was built on.
const BEHAVIOUR = [
  ["Power Users", 2.81, 0.129, 0.048, 4949],
  ["Premium Engagers", 2.73, 0.131, 0.053, 4829],
  ["Digital Natives", 2.75, 0.126, 0.050, 4986],
  ["Casual Browsers", 2.75, 0.122, 0.049, 5071],
];

export default function Finding() {
  return (
    <Panel className="finding" aria-labelledby="finding-heading">
      <p className="eyebrow">Where this came from</p>
      <h2 id="finding-heading">Four segments that behaved identically</h2>

      <div className="finding-grid">
        <div className="finding-text">
          <p>
            This started as a segmentation dashboard. It clustered a thousand
            ad-audience profiles into four groups and gave them names: Power
            Users, Premium Engagers, Digital Natives, Casual Browsers. The
            names were assigned by cluster index, which k-means hands out
            arbitrarily, and were never checked against what the clusters did.
          </p>
          <p>
            Checked, they are indistinguishable. The Power Users spend three
            minutes a day longer online than the Casual Browsers. The Premium
            Engagers have the <i>fewest</i> likes of any group.
          </p>
          <p>
            The data cannot support segments. Every categorical field is close
            to uniform, the largest correlation between any two behavioural
            columns is 0.08, and click-through rate does not differ by age,
            gender, income or education at any conventional threshold. The
            columns were generated independently, so there is nothing joint to
            find.
          </p>
          <p>
            The training script did compute an elbow curve over k from 1 to 10.
            It then set <code>optimal_k = 4</code> without looking at it. The
            curve has no elbow: inertia falls smoothly and evenly the whole way.
          </p>
        </div>

        <figure className="finding-table">
          <figcaption>The four segments, as measured</figcaption>
          <div className="bt-head">
            <span>Segment</span><span>hrs/day</span><span>CTR</span><span>conv.</span><span>likes</span>
          </div>
          <ul>
            {BEHAVIOUR.map(([name, hrs, ctr, conv, likes]) => (
              <li key={name}>
                <span className="bt-name">{name}</span>
                <span>{hrs.toFixed(2)}</span>
                <span>{ctr.toFixed(3)}</span>
                <span>{conv.toFixed(3)}</span>
                <span>{likes.toLocaleString()}</span>
              </li>
            ))}
          </ul>
          <p className="bt-foot">
            The whole spread across four "distinct" segments: 0.08 hours a day,
            0.009 of CTR, 0.005 of conversion. Naming them was the error, not
            the clustering: k-means did what it was asked. Nobody asked whether
            there was anything to cluster.
          </p>
        </figure>
      </div>
    </Panel>
  );
}
