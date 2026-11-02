"""Shared helpers for the SAM-passivation pipeline: paths, RDKit features, dataset I/O.

Every script imports ROOT from here, so the repository runs after `git clone`
on any machine (no absolute paths are stored anywhere).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")
QC = os.path.join(ROOT, "qc")
for _d in (OUT, FIG):
    os.makedirs(_d, exist_ok=True)

# ---- dataset schema ----------------------------------------------------------------
REQUIRED = ["Molecule", "SMILES"]
# Control_PCE / Modified_PCE allow the Su-style target "Improved_Ratio";
# if Control_PCE is absent the pipeline falls back to regressing Modified_PCE itself.
COLS = ["Molecule", "Abbrev", "SMILES", "Anchor", "Device_Structure", "Processing",
        "Control_PCE", "Modified_PCE", "Improved_Ratio", "Voc", "Jsc", "FF",
        "Source", "DOI"]

# ---- anchor groups that can bind ITO / NiOx / perovskite Pb --------------------------
ANCHOR_SMARTS = {
    "PO3H2": "P(=O)([OH])[OH]",
    "COOH": "[CX3](=O)[OX2H1]",
    "SO3H": "S(=O)(=O)[OH]",
}

# ---- physical descriptors (interpretable, added to the MACCS block) -------------------
PHYSCHEM = ["MolWt", "NumAromaticRings", "NumRotatableBonds", "TPSA", "MolLogP",
            "NumHDonors", "NumHAcceptors", "FractionCSP3", "HeavyAtoms", "RingCount"]


def read_dataset(name="dataset"):
    """Accept data/<name>.xlsx or data/<name>.csv; return a DataFrame with COLS present."""
    for ext in ("xlsx", "csv"):
        path = os.path.join(DATA, f"{name}.{ext}")
        if os.path.exists(path):
            df = pd.read_excel(path) if ext == "xlsx" else pd.read_csv(path)
            break
    else:
        raise SystemExit(f"dataset not found under {DATA}: {name}.xlsx / {name}.csv")
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def target_name(df):
    """Su-style improvement ratio when paired device data exist, else absolute PCE."""
    ratio = pd.to_numeric(df.get("Improved_Ratio"), errors="coerce")
    if ratio is not None and ratio.notna().sum() >= 3:
        return "Improved_Ratio"
    return "Modified_PCE"


def save_features(features, sparse_limit, path=None):
    path = path or os.path.join(OUT, "features.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"features": features, "sparse_limit": sparse_limit}, f, indent=2)


def load_features(path=None):
    path = path or os.path.join(OUT, "features.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["features"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))
