#!/usr/bin/env python3
"""
analysis_binding_site.py

Binding-site residue composition analysis: identifies protein residues in
contact with the ligand (as originally determined per-complex by PLIP) and
summarizes their physicochemical class distribution and per-residue contact
occupancy across an MD trajectory.

Author: Linta Mahboob
Requires: MDAnalysis >= 2.4, numpy, matplotlib, pandas
Usage:
    python analysis_binding_site.py --tpr md.tpr --xtc md_noPBC.xtc --ligand-resname LIG
"""

import argparse
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required: pip install pandas")

try:
    import MDAnalysis as mda
except ImportError:
    raise ImportError("MDAnalysis is required: pip install MDAnalysis")

RESIDUE_CLASS = {
    "ASP": "Charged (acidic)", "GLU": "Charged (acidic)",
    "ARG": "Charged (basic)", "LYS": "Charged (basic)", "HIS": "Charged (basic)",
    "SER": "Polar", "THR": "Polar", "ASN": "Polar", "GLN": "Polar",
    "TYR": "Polar", "CYS": "Polar",
    "ALA": "Hydrophobic", "VAL": "Hydrophobic", "LEU": "Hydrophobic",
    "ILE": "Hydrophobic", "PRO": "Hydrophobic", "PHE": "Hydrophobic",
    "MET": "Hydrophobic", "TRP": "Hydrophobic",
    "GLY": "Special",
}

CLASS_COLORS = {
    "Charged (acidic)": "#C0392B",
    "Charged (basic)": "#8E44AD",
    "Polar": "#2980B9",
    "Hydrophobic": "#E67E22",
    "Special": "#7F8C8D",
}

CONTACT_CUTOFF = 5.0  # A


def parse_args():
    p = argparse.ArgumentParser(description="Binding-site residue composition analysis")
    p.add_argument("--tpr", default="md.tpr")
    p.add_argument("--xtc", default="md_noPBC.xtc")
    p.add_argument("--ligand-resname", default="LIG")
    p.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF)
    p.add_argument("--outdir", default="analysis_output")
    p.add_argument("--every-nth-frame", type=int, default=10)
    return p.parse_args()


def collect_binding_site_residues(universe, ligand_resname, cutoff, step):
    """Protein residues within `cutoff` A of the ligand, sampled every `step` frames."""
    lig = universe.select_atoms(f"resname {ligand_resname}")
    if len(lig) == 0:
        raise ValueError(f"Ligand '{ligand_resname}' not found in topology.")

    all_contacts = []
    contact_freq = Counter()
    n_sampled = 0

    for ts in universe.trajectory[::step]:
        shell = universe.select_atoms(f"protein and around {cutoff} resname {ligand_resname}")
        seen = set()
        for atom in shell:
            all_contacts.append(atom.resname)
            key = (atom.resname, atom.resid)
            if key not in seen:
                contact_freq[key] += 1
                seen.add(key)
        n_sampled += 1

    return all_contacts, contact_freq, n_sampled


def compute_frequency_table(all_contacts, contact_freq, n_sampled):
    total = len(all_contacts)
    restype_count = Counter(all_contacts)

    restype_df = pd.DataFrame(
        [(k, v, v / total * 100) for k, v in restype_count.most_common()],
        columns=["resname", "count", "percentage"],
    )
    restype_df["class"] = restype_df["resname"].map(lambda r: RESIDUE_CLASS.get(r, "Other"))

    occ_rows = [(rn, rid, cnt, cnt / n_sampled * 100) for (rn, rid), cnt in contact_freq.items()]
    occ_df = pd.DataFrame(occ_rows, columns=["resname", "resid", "frames", "occupancy_pct"])
    occ_df = occ_df.sort_values("occupancy_pct", ascending=False).reset_index(drop=True)

    charged_pct = restype_df.loc[restype_df["class"].str.startswith("Charged"), "percentage"].sum()
    print(f"Charged residues in binding site: {charged_pct:.1f}%")

    return restype_df, occ_df


def plot_residue_composition(restype_df, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(wspace=0.4)

    class_totals = (
        restype_df.groupby("class")["percentage"].sum().reset_index()
        .sort_values("percentage", ascending=False)
    )
    colors = [CLASS_COLORS.get(c, "#BDC3C7") for c in class_totals["class"]]
    ax1.pie(class_totals["percentage"], labels=class_totals["class"], autopct="%1.1f%%",
            colors=colors, startangle=140, pctdistance=0.78,
            wedgeprops={"linewidth": 1, "edgecolor": "white"})
    ax1.set_title("Binding-site residue classes")

    top_res = restype_df.head(15)
    bar_colors = [CLASS_COLORS.get(c, "#BDC3C7") for c in top_res["class"]]
    ax2.barh(top_res["resname"][::-1], top_res["percentage"][::-1],
              color=bar_colors[::-1], edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Frequency in binding site (%)")
    ax2.set_title("Top binding-site residues")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Binding-Site Residue Composition", fontsize=12, y=1.02)
    out_path = os.path.join(outdir, "binding_site_residue_composition.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_contact_occupancy(occ_df, outdir, top_n=30):
    top = occ_df.head(top_n).copy()
    colors = [CLASS_COLORS.get(RESIDUE_CLASS.get(rn.upper(), ""), "#5D8AA8") for rn in top["resname"]]
    labels = top.apply(lambda r: f"{r['resname']}{r['resid']}", axis=1)

    fig, ax = plt.subplots(figsize=(max(9, len(top) * 0.45), 4.5))
    ax.bar(range(len(top)), top["occupancy_pct"], color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.0, label="50% occupancy")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Contact occupancy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-residue binding-site contact occupancy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)

    out_path = os.path.join(outdir, "binding_site_contact_occupancy.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.tpr) or not os.path.exists(args.xtc):
        raise FileNotFoundError(f"Missing input files: {args.tpr}, {args.xtc}")

    u = mda.Universe(args.tpr, args.xtc)
    print(f"Loaded {len(u.atoms)} atoms, {len(u.trajectory)} frames")

    all_contacts, contact_freq, n_sampled = collect_binding_site_residues(
        u, args.ligand_resname, args.cutoff, args.every_nth_frame
    )
    if not all_contacts:
        print("No binding-site residues found; check ligand resname and cutoff.")
        return

    restype_df, occ_df = compute_frequency_table(all_contacts, contact_freq, n_sampled)

    restype_df.to_csv(os.path.join(args.outdir, "residue_type_frequency.csv"), index=False, float_format="%.3f")
    occ_df.to_csv(os.path.join(args.outdir, "residue_contact_occupancy.csv"), index=False, float_format="%.3f")

    plot_residue_composition(restype_df, args.outdir)
    plot_contact_occupancy(occ_df, args.outdir)

    print(f"All outputs written to {args.outdir}/")


if __name__ == "__main__":
    main()
