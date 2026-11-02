"""
Step 5 - parse the QC outputs and rank the SAM shortlist.

Expected layout (produced by step4_qc_inputs.py):

    qc/gaussian/Pb2I4.log                       bare cluster
    qc/gaussian/<name>_monomer.log              isolated SAM
    qc/gaussian/<name>_Pb2I4.log                anchored SAM
    qc/vasp/niO_clean/OUTCAR                    oxide reference
    qc/vasp/niO_<name>/OUTCAR                   SAM on NiO(100)
    qc/vasp/pbi2_clean/OUTCAR                   PbI2 reference
    qc/vasp/pbi2_<name>/OUTCAR                  SAM on the PbI2-terminated perovskite model

Descriptors (eV unless noted):
    dE_anchor   = E(SAM+Pb2I4) - E(SAM) - E(Pb2I4)      Pb-side binding strength
    Eads_oxide  = E(NiO+SAM) - E(NiO) - E(SAM_gas)      oxide-side chemisorption
    Eads_perov  = E(PbI2+SAM) - E(PbI2) - E(SAM_gas)
    gap         = HOMO-LUMO gap of the isolated molecule
    dipole_D    = molecular dipole moment (Debye); a larger normal dipole shifts the
                  electrode work function and helps hole extraction
    score       = 0.5*norm(-Eads_oxide) + 0.3*norm(-dE_anchor) + 0.2*norm(|dipole_D|)

Usage:
    python src/step5_analyze_qc.py [--qc qc]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, QC  # noqa: E402

HARTREE2EV = 27.211386


def gaussian_energy(log):
    if not os.path.exists(log):
        return np.nan
    e = np.nan
    with open(log, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "SCF Done" in line:
                e = float(line.split("=")[1].split()[0])
    return e


def gaussian_gap(log):
    if not os.path.exists(log):
        return np.nan
    occ = vir = None
    with open(log, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"Alpha\s+occ\. eigenvalues\s+--\s+(\S+)\s+(\S+)", line)
            if m:
                occ = float(m.group(2))
            m = re.search(r"Alpha\s+virt\. eigenvalues\s+--\s+(\S+)\s+(\S+)", line)
            if m:
                vir = float(m.group(1))
    return (vir - occ) * HARTREE2EV if occ is not None and vir is not None else np.nan


def gaussian_dipole(log):
    if not os.path.exists(log):
        return np.nan
    pending = False
    with open(log, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Dipole moment (field-independent basis, Debye)" in line:
                if "Tot=" in line:          # some builds print everything on one line
                    return float(line.split("Tot=")[1].split()[0])
                pending = True             # Gaussian prints "Tot=" on the next line
                continue
            if pending and "Tot=" in line:
                return float(line.split("Tot=")[1].split()[0])
    return np.nan


def vasp_energy(outcar):
    if not os.path.exists(outcar):
        return np.nan
    e = np.nan
    with open(outcar, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                e = float(line.split("=")[1].split()[0])
    return e


def norm(s, invert=True):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0 or s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    v = (s - s.min()) / (s.max() - s.min())
    return 1 - v if invert else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", default=QC)
    args = ap.parse_args()
    gau, vas = os.path.join(args.qc, "gaussian"), os.path.join(args.qc, "vasp")

    sel = os.path.join(OUT, "qc_candidates.csv")
    if os.path.exists(sel):
        names = list(pd.read_csv(sel)["name"])
        # keep only shortlisted molecules whose monomer log has actually been run
        names = [n for n in names
                 if os.path.exists(os.path.join(gau, f"{n}_monomer.log"))]
    else:
        names = []
    if not names and os.path.isdir(gau):
        names = sorted(f[:-11] for f in os.listdir(gau) if f.endswith("_monomer.log"))
    if not names:
        print("no QC outputs found yet - inputs are ready, run them on a licensed cluster")
        return

    e_cluster = gaussian_energy(os.path.join(gau, "Pb2I4.log"))
    e_oxide = vasp_energy(os.path.join(vas, "niO_clean", "OUTCAR"))
    e_perov = vasp_energy(os.path.join(vas, "pbi2_clean", "OUTCAR"))

    rows = []
    for name in names:
        mono = os.path.join(gau, f"{name}_monomer.log")
        e_mono = gaussian_energy(mono)
        e_cx = gaussian_energy(os.path.join(gau, f"{name}_Pb2I4.log"))
        rows.append({
            "molecule": name,
            "E_monomer_eV": e_mono * HARTREE2EV if np.isfinite(e_mono) else np.nan,
            "gap_eV": gaussian_gap(mono),
            "dipole_D": gaussian_dipole(mono),
            "dE_anchor": (e_cx - e_mono - e_cluster) * HARTREE2EV
            if all(np.isfinite([e_cx, e_mono, e_cluster])) else np.nan,
            "Eads_oxide": (vasp_energy(os.path.join(vas, f"niO_{name}", "OUTCAR"))
                           - e_oxide - e_mono * HARTREE2EV)
            if np.isfinite([e_oxide, e_mono]).all() else np.nan,
            "Eads_perov": (vasp_energy(os.path.join(vas, f"pbi2_{name}", "OUTCAR"))
                           - e_perov - e_mono * HARTREE2EV)
            if np.isfinite([e_perov, e_mono]).all() else np.nan,
        })

    df = pd.DataFrame(rows)
    df["score"] = (0.5 * norm(df["Eads_oxide"]) + 0.3 * norm(df["dE_anchor"])
                   + 0.2 * norm(df["dipole_D"].abs(), invert=False))
    df = df.sort_values("score", ascending=False)
    out = os.path.join(OUT, "qc_descriptors.csv")
    df.to_csv(out, index=False)

    print(df.round(3).to_string(index=False))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
