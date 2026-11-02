"""
Step 4 - quantum-chemistry and DFT inputs for the SAM shortlist.

Gaussian (B3LYP + D3(BJ); C/H/N/O/P = 6-31G(d,p), I/Pb = Def2TZVP; pop=MK):
    * <name>_monomer.gjf        geometry + frequencies + ESP charges + dipole
    * <name>_Pb2I4.gjf          phosphonate anchored on a Pb2I4 surface fragment
    * Pb2I4.gjf                 bare cluster reference for the binding energy
VASP (NiO(100) 3x3x3 slab / FAPbI3(001) 2x2x2 PbI2 slab, 15 A vacuum, 520 eV, 2x2x1):
    * niO_clean, fapbi3_clean                  references
    * niO_<name>, fapbi3_<name>                SAM standing on the surface, anchor down

Why these two surfaces: a SAM must do two jobs at once - chemisorb on the oxide
electrode (ITO/NiOx) and passivate the perovskite Pb sites. The Gaussian cluster gives
the Pb-side binding energy, the VASP slabs give the oxide-side adsorption energy and the
standing geometry; both are needed before a shortlist can be ranked.

Outputs: qc/gaussian/*.gjf, qc/vasp/<case>/{INCAR,KPOINTS,POSCAR,run.sh,adsorbate_indices.txt}

Usage:
    python src/step4_qc_inputs.py [--top 12]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, QC  # noqa: E402

from ase import Atoms
from ase.io import write
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
GAU = os.path.join(QC, "gaussian")
VAS = os.path.join(QC, "vasp")
for _d in (GAU, VAS):
    os.makedirs(_d, exist_ok=True)

HEADER = """%mem=8GB
%nprocshared=8
#p B3LYP/GENECP em=GD3BJ {route} int=ultrafine pop=MK iop(6/33=2) scf=xqc

{title}

{charge} {mult}
"""
ROUTE_MONO = "opt freq"
ROUTE_CLUSTER = "opt=(calcfc,maxstep=8,maxcycles=300) freq"
LIGHT = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br"]
HEAVY = ["Pb", "I"]
LIGHT_BASIS = "6-31G(d,p)"
HEAVY_BASIS = "Def2TZVP"


def basis_block(symbols):
    have = {s for s in symbols}
    light = [s for s in LIGHT if s in have]
    heavy = [s for s in HEAVY if s in have]
    out = ""
    if light:
        out += " ".join(light) + " 0\n" + LIGHT_BASIS + "\n****\n"
    if heavy:
        out += " ".join(heavy) + " 0\n" + HEAVY_BASIS + "\n****\n"
    return out


def write_gjf(path, title, symbols, xyz, charge=0, mult=1, route=ROUTE_MONO):
    body = "".join(f"{s:<3s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}\n"
                   for s, p in zip(symbols, xyz))
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER.format(title=title, charge=charge, mult=mult, route=route))
        f.write(body)
        # GENECP card order: basis block -> blank (ends basis) -> blank (empty ECP section)
        f.write("\n" + basis_block(symbols) + "\n\n\n")


def embed(smiles, seed=0xf00d):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def coords(mol):
    c = mol.GetConformer()
    return np.array([list(c.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def anchor_atom(mol):
    """Return (P index, binding-O index): the P=O oxygen that coordinates the metal."""
    p_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P"][0]
    o_idx = [n.GetIdx() for n in mol.GetAtomWithIdx(p_idx).GetNeighbors()
             if n.GetSymbol() == "O"]
    double = [i for i in o_idx if mol.GetBondBetweenAtoms(p_idx, i).GetBondTypeAsDouble() == 2]
    return p_idx, (double[0] if double else o_idx[0])


def align(vec, target):
    """Rotation matrix that maps `vec` onto `target` (Rodrigues)."""
    a = np.asarray(vec, float) / np.linalg.norm(vec)
    b = np.asarray(target, float) / np.linalg.norm(target)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / np.linalg.norm(v) ** 2)


def oriented_molecule(mol):
    """Molecule rotated so that P=O points along -z, centred on that oxygen."""
    p, o = anchor_atom(mol)
    xyz = coords(mol)
    R = align(xyz[o] - xyz[p], np.array([0.0, 0.0, -1.0]))
    xyz = xyz @ R.T
    return [a.GetSymbol() for a in mol.GetAtoms()], xyz - xyz[o], o


# ---------------------------------------------------------------------------- Gaussian
PB2I4 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.2],
                  [2.2, 0.0, -0.9], [-2.2, 0.0, -0.9],
                  [2.2, 0.0, 3.9], [-2.2, 0.0, 3.9]])
PB2I4_SYM = ["Pb", "Pb", "I", "I", "I", "I"]


def gaussian_inputs(names):
    made = 0
    write_gjf(os.path.join(GAU, "Pb2I4.gjf"),
              "Pb2I4 cluster (neutral, singlet): perovskite-surface fragment reference",
              PB2I4_SYM, PB2I4)
    made += 1
    for name, smi in names:
        mol = embed(smi)
        sym, xyz = [a.GetSymbol() for a in mol.GetAtoms()], coords(mol)
        write_gjf(os.path.join(GAU, f"{name}_monomer.gjf"),
                  f"{name}: geometry + ESP + dipole (B3LYP-GD3BJ)", sym, xyz)
        made += 1

        # anchor-down complex: P=O oxygen 2.35 A above the top Pb of the cluster
        msym, mxyz, o = oriented_molecule(mol)
        top_pb = PB2I4[1]
        shift = np.array([0.0, 0.0, top_pb[2] + 2.35])
        cx_sym = PB2I4_SYM + msym
        cx_xyz = np.vstack([PB2I4, mxyz + shift])
        write_gjf(os.path.join(GAU, f"{name}_Pb2I4.gjf"),
                  f"{name} anchored on Pb2I4 (neutral, singlet)",
                  cx_sym, cx_xyz, route=ROUTE_CLUSTER)
        made += 1
    print(f"Gaussian: {made} inputs -> {GAU}")


# ---------------------------------------------------------------------------- VASP
NI0_A = 4.177          # rock-salt NiO cubic lattice constant (Angstrom)
PEROV_A = 6.362        # FAPbI3 pseudo-cubic lattice constant (paper SI)
VACUUM = 15.0
INCAR = """SYSTEM = SAM on oxide / perovskite
PREC   = Accurate
ENCUT  = 520
EDIFF  = 1E-5
EDIFFG = -0.03
ISMEAR = 0
SIGMA  = 0.05
ISIF   = 2
IBRION = 2
NSW    = 300
LREAL  = .FALSE.
LWAVE  = .FALSE.
LCHARG = .FALSE.
LASPH  = .TRUE.
IVDW   = 12
"""
KPOINTS = """Automatic
0
Gamma
2 2 1
0 0 0
"""
RUN = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
mpirun -np 64 vasp_std > vasp.out
"""


def nio_slab(nx=3, ny=3, layers=3):
    """Rock-salt NiO(100): 4 Ni + 4 O per cubic cell, NiO stoichiometry."""
    syms, pos = [], []
    base = [("Ni", (0.0, 0.0, 0.0)), ("Ni", (0.0, 0.5, 0.5)),
            ("Ni", (0.5, 0.0, 0.5)), ("Ni", (0.5, 0.5, 0.0)),
            ("O", (0.5, 0.0, 0.0)), ("O", (0.0, 0.5, 0.0)),
            ("O", (0.0, 0.0, 0.5)), ("O", (0.5, 0.5, 0.5))]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(layers):
                for s, f in base:
                    syms.append(s)
                    pos.append([(f[0] + ix) * NI0_A, (f[1] + iy) * NI0_A, (f[2] + iz) * NI0_A])
    return syms, np.array(pos)


def pbi2_slab(nx=2, ny=2, layers=2):
    """PbI2-terminated FAPbI3(001) model: one Pb + two I per site, neutral PbI2 stoichiometry.
    A-site cations are omitted for cost (documented in README)."""
    syms, pos = [], []
    base = [("Pb", (0.5, 0.5, 0.5)), ("I", (0.5, 0.5, 0.0)), ("I", (0.5, 0.0, 0.5))]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(layers):
                for s, f in base:
                    syms.append(s)
                    pos.append([(f[0] + ix) * PEROV_A, (f[1] + iy) * PEROV_A,
                                (f[2] + iz) * PEROV_A])
    return syms, np.array(pos)


def place_on_surface(slab_syms, slab_pos, mol, site_symbol, gap, lx, ly):
    """Stand the molecule on the topmost `site_symbol` atom, anchor oxygen pointing down."""
    msym, mxyz, _ = oriented_molecule(mol)
    zmax = slab_pos[:, 2].max()
    idx = [i for i, s in enumerate(slab_syms)
           if s == site_symbol and abs(slab_pos[i, 2] - zmax) < 0.6]
    site = slab_pos[idx[0]]
    mxyz = mxyz + site + np.array([0.0, 0.0, gap])
    mxyz[:, 0] %= lx          # keep the molecule inside the periodic cell
    mxyz[:, 1] %= ly
    return slab_syms + msym, np.vstack([slab_pos, mxyz])


def write_vasp(case, syms, pos, lx, ly):
    d = os.path.join(VAS, case)
    os.makedirs(d, exist_ok=True)
    pos = pos.copy()
    pos[:, 2] = pos[:, 2] - pos[:, 2].min()
    cell = np.array([[lx, 0, 0], [0, ly, 0],
                     [0, 0, pos[:, 2].max() + VACUUM]])
    atoms = Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)
    # sort=True groups the species so that the POSCAR has standard unique-symbol lines
    write(os.path.join(d, "POSCAR"), atoms, format="vasp", direct=False, sort=True)
    with open(os.path.join(d, "INCAR"), "w", encoding="utf-8") as f:
        f.write(INCAR)
    with open(os.path.join(d, "KPOINTS"), "w", encoding="utf-8") as f:
        f.write(KPOINTS)
    with open(os.path.join(d, "run.sh"), "w", encoding="utf-8") as f:
        f.write(RUN)


def vasp_inputs(names):
    usym, upos = nio_slab()
    psym, ppos = pbi2_slab()
    write_vasp("niO_clean", usym, upos, 3 * NI0_A, 3 * NI0_A)
    write_vasp("pbi2_clean", psym, ppos, 2 * PEROV_A, 2 * PEROV_A)
    for name, smi in names:
        mol = embed(smi)
        write_vasp(f"niO_{name}", *place_on_surface(usym, upos, mol, "Ni", 2.10,
                                                    3 * NI0_A, 3 * NI0_A),
                   lx=3 * NI0_A, ly=3 * NI0_A)
        write_vasp(f"pbi2_{name}", *place_on_surface(psym, ppos, mol, "Pb", 2.40,
                                                    2 * PEROV_A, 2 * PEROV_A),
                   lx=2 * PEROV_A, ly=2 * PEROV_A)
    print(f"VASP: {2 + 2 * len(names)} cases -> {VAS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    sel = os.path.join(OUT, "qc_candidates.csv")
    if os.path.exists(sel):
        df = pd.read_csv(sel).head(args.top)
        names = list(zip(df["name"], df["SMILES"]))
    else:
        df = pd.read_csv(os.path.join(DATA, "candidates.csv")).head(args.top)
        names = list(zip(df["name"], df["SMILES"]))
    gaussian_inputs(names)
    vasp_inputs(names)


if __name__ == "__main__":
    main()
