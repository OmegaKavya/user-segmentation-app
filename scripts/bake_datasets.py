"""Prepare the datasets the page ships with.

Two of them, on purpose. A test that only ever says "no structure" is
indistinguishable from a broken test, so the page carries one dataset where
the answer is no and one where it is yes, and runs the identical procedure on
both in front of the reader.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

CATEGORICAL = ["Age", "Gender", "Location", "Language",
               "Education Level", "Device Usage", "Income Level"]


def ad_audience():
    """The original dataset, encoded the way the original project encoded it."""
    df = pd.read_csv(os.path.join(ROOT, "data", "user_profiles_for_ads.csv"))
    df = df.drop("User ID", axis=1)
    for col in CATEGORICAL:
        df[col] = LabelEncoder().fit_transform(df[col])
    df["Top Interests"] = df["Top Interests"].fillna("")
    mlb = MultiLabelBinarizer()
    interests = pd.DataFrame(
        mlb.fit_transform(df["Top Interests"].str.split(", ")), columns=mlb.classes_
    )
    processed = pd.concat([df.drop("Top Interests", axis=1), interests], axis=1)
    return StandardScaler().fit_transform(processed), list(processed.columns)


def customer_tiers():
    """Three genuine groups, with enough overlap to be realistic."""
    X, _ = make_blobs(n_samples=900, n_features=8, centers=3,
                      cluster_std=2.6, random_state=0)
    names = ["spend", "visits", "basket size", "recency", "tenure",
             "support tickets", "discount use", "returns"]
    return StandardScaler().fit_transform(X), names


def main():
    out = {}
    for key, title, note, builder in [
        ("ads", "Ad audience, 1000 users",
         "The dataset this project was built on. Demographics, interests and "
         "engagement, encoded exactly as the original pipeline encoded them.",
         ad_audience),
        ("tiers", "Customer tiers, 900 records",
         "Constructed to contain three groups, as a control. The same "
         "procedure has to find them, or it is not measuring anything.",
         customer_tiers),
    ]:
        X, columns = builder()
        out[key] = {
            "title": title,
            "note": note,
            "columns": columns,
            # Four decimals is far finer than anything the result depends on
            # and roughly halves the payload.
            "X": [[round(float(v), 4) for v in row] for row in X],
        }
        print(f"{key}: {len(X)} rows x {len(columns)} features")

    path = os.path.join(ROOT, "web", "src", "datasets.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle, separators=(",", ":"))
    print(f"wrote {os.path.getsize(path) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
