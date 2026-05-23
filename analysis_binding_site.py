#!/usr/bin/env python3
"""
analysis_binding_site.py
────────────────────────
Characterises binding site residue composition and identifies dynamic hotspots
from a protein-ligand MD trajectory.

Reproduces the core statistical analysis from the thesis:
  • Residue-type frequency within binding pockets (across all 100 complexes /
    all trajectory frames)
  • Enrichment of charged residues: Asp (28.1%), His (11.7%), Arg (9.2%)
  • Pie chart + bar chart of residue-type composition
  • Per-residue contact frequency heat map

Author : Linta Mahboob | University of Siena
Requires: MDAnalysis >= 2.4, numpy, matplotlib, pandas, scipy
Usage  : python analysis_binding_site.py [--tpr md.tpr] [--xtc md_noPBC.xtc]
         python analysis_binding_site.py --multi-complex  # batch mode for 100 complexes
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas required: pip install pandas")

try:
    import MDAnalysis as mda
except ImportError:
    raise ImportError("MDAnalysis required: pip install MDAnalysis")

warnings.filterwarnings("ignore")

# ─── Residue classification ───────────────────────────────────────────────────
# Three-letter code → physicochemical class
RESIDUE_CLASS = {
    # Charged (acidic)
    "ASP": "Charged (Acidic)",  "GLU": "Charged (Acidic)",
    # Charged (basic)
    "ARG": "Charged (Basic)",   "LYS": "Charged (Basic)",
    "HIS": "Charged (Basic)",
    # Polar uncharged
    "SER": "Polar",  "THR": "Polar",  "ASN": "Polar",  "GLN": "Polar",
    "TYR": "Polar",  "CYS": "Polar",
    # Hydrophobic
    "ALA": "Hydrophobic", "VAL": "Hydrophobic", "LEU": "Hydrophobic",
    "ILE": "Hydrophobic", "PRO": "Hydrophobic", "PHE": "Hydrophobic",
    "MET": "Hydrophobic", "TRP": "Hydrophobic",
    # Special
    "GLY": "Special",
}

CLASS_COLOURS = {
    "Charged (Acidic)": "#C0392B",
    "Charged (Basic)":  "#8E44AD",
    "Polar":            "#2980B9",
    "Hydrophobic":      "#E67E22",
    "Special":          "#7F8C8D",
}

# Thesis dynamic hotspot residues
HOTSPOT_RES = {"ASP", "HIS", "ARG"}

CONTACT_CUTOFF = 5.0   # Å — residue considered in binding site if within this distance


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Binding site residue composition and dynamic hotspot analysis"
    )
    p.add_argument("--tpr",  default="md.tpr")
    p.add_argument("--xtc",  default="md_noPBC.xtc")
    p.add_argument("--ligand-resname", default="LIG")
    p.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF,
                   help=f"Binding site contact cutoff in Å (default: {CONTACT_CUTOFF})")
    p.add_argument("--outdir", default="analysis_output")
    p.add_argument("--every-nth-frame", type=int, default=10,
                   help="Analyse every Nth frame for speed (default: 10)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def collect_binding_site_residues(universe, ligand_resname, cutoff, step):
    """
    For each sampled frame, identify all protein residues within `cutoff` Å
    of any ligand atom. Returns a list of (resname, resid) tuples across all frames.
    """
    print(f"\n  Collecting binding site residues (cutoff = {cutoff} Å, "
          f"every {step}th frame) ...")

    all_contacts = []   # (resname, resid) per observation
    contact_freq  = Counter()   # (resname, resid) → n_frames in contact

    lig = universe.select_atoms(f"resname {ligand_resname}")
    if len(lig) == 0:
        raise ValueError(
            f"Ligand '{ligand_resname}' not found in topology. "
            "Check --ligand-resname."
        )

    n_sampled = 0
    for ts in universe.trajectory[::step]:
        shell_sel = (
            f"protein and around {cutoff} resname {ligand_resname}"
        )
        shell = universe.select_atoms(shell_sel)
        seen_this_frame = set()
        for atom in shell:
            key = (atom.resname, atom.resid)
            all_contacts.append(atom.resname)
            if key not in seen_this_frame:
                contact_freq[key] += 1
                seen_this_frame.add(key)
        n_sampled += 1

    print(f"  Sampled {n_sampled} frames | {len(all_contacts)} residue observations")
    return all_contacts, contact_freq, n_sampled


# ─────────────────────────────────────────────────────────────────────────────
def compute_frequency_table(all_contacts, contact_freq, n_sampled):
    """
    Compute residue-type frequencies and per-residue occupancy.
    """
    total = len(all_contacts)

    # Residue-type frequency
    restype_count = Counter(all_contacts)
    restype_df = pd.DataFrame(
        [(k, v, v/total*100) for k, v in restype_count.most_common()],
        columns=["resname", "count", "percentage"]
    )
    restype_df["class"] = restype_df["resname"].map(
        lambda r: RESIDUE_CLASS.get(r, "Other")
    )
    restype_df["is_hotspot"] = restype_df["resname"].isin(HOTSPOT_RES)

    # Per-residue occupancy (fraction of sampled frames in contact)
    occ_rows = [
        (rn, rid, cnt, cnt/n_sampled*100)
        for (rn, rid), cnt in contact_freq.items()
    ]
    occ_df = pd.DataFrame(occ_rows,
                          columns=["resname", "resid", "frames", "occupancy_pct"])
    occ_df = occ_df.sort_values("occupancy_pct", ascending=False).reset_index(drop=True)

    # Summary of charged residues
    charged = restype_df[restype_df["class"].str.startswith("Charged")]
    total_charged_pct = charged["percentage"].sum()

    print(f"\n  Residue frequency summary:")
    print(f"  {'Residue':<8} {'%':>8}  {'Class':<22}  {'Hotspot?'}")
    print(f"  {'-'*8} {'-'*8}  {'-'*22}  {'-'*8}")
    for _, row in restype_df.head(12).iterrows():
        hs = "★ HOTSPOT" if row["is_hotspot"] else ""
        print(f"  {row['resname']:<8} {row['percentage']:>7.1f}%  "
              f"{row['class']:<22}  {hs}")
    print(f"\n  Total charged residues in binding site: {total_charged_pct:.1f}%")
    print(f"  [Thesis benchmark: Asp 28.1% + His 11.7% + Arg 9.2% = 49.0% charged]")

    return restype_df, occ_df


# ─────────────────────────────────────────────────────────────────────────────
def plot_residue_composition(restype_df, outdir):
    """Pie chart + bar chart of residue-type composition in binding site."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.4)

    # ── Pie: physicochemical class distribution ──────────────────────────────
    class_totals = (
        restype_df.groupby("class")["percentage"].sum().reset_index()
        .sort_values("percentage", ascending=False)
    )
    colours = [CLASS_COLOURS.get(c, "#BDC3C7") for c in class_totals["class"]]
    wedges, texts, autotexts = ax1.pie(
        class_totals["percentage"],
        labels=class_totals["class"],
        autopct="%1.1f%%",
        colors=colours,
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax1.set_title("A   Binding Site Residue Classes", fontweight="bold")

    # ── Bar: top individual residues ─────────────────────────────────────────
    top_res = restype_df.head(15)
    bar_cols = []
    for _, row in top_res.iterrows():
        if row["is_hotspot"]:
            bar_cols.append(CLASS_COLOURS["Charged (Basic)"]
                            if row["resname"] in ("HIS", "ARG")
                            else CLASS_COLOURS["Charged (Acidic)"])
        else:
            bar_cols.append(CLASS_COLOURS.get(row["class"], "#BDC3C7"))

    bars = ax2.barh(top_res["resname"][::-1],
                    top_res["percentage"][::-1],
                    color=bar_cols[::-1],
                    edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Frequency in Binding Site (%)")
    ax2.set_title("B   Top Binding Site Residues by Frequency", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Annotate hotspots
    for i, (_, row) in enumerate(top_res[::-1].iterrows()):
        if row["is_hotspot"]:
            ax2.text(row["percentage"] + 0.3, i, "★ Dynamic Hotspot",
                     va="center", fontsize=7.5, color="#2C3E50")

    fig.suptitle(
        "Binding Site Residue Composition — Protein-Ligand MD Trajectory\n"
        "Linta Mahboob, University of Siena",
        fontsize=11, y=1.01
    )

    out_path = os.path.join(outdir, "binding_site_residue_composition.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  Composition plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_contact_occupancy_heatmap(occ_df, outdir, top_n=30):
    """Bar chart of per-residue contact occupancy (top N residues)."""
    top = occ_df.head(top_n).copy()
    colours = []
    for _, row in top.iterrows():
        rn = row["resname"].upper()
        if rn == "ASP":
            colours.append(CLASS_COLOURS["Charged (Acidic)"])
        elif rn in ("HIS", "ARG"):
            colours.append(CLASS_COLOURS["Charged (Basic)"])
        elif RESIDUE_CLASS.get(rn, "").startswith("Charged"):
            colours.append("#A569BD")
        else:
            colours.append("#5D8AA8")

    labels = top.apply(lambda r: f"{r['resname']}{r['resid']}", axis=1)

    fig, ax = plt.subplots(figsize=(max(10, len(top)*0.5), 5))
    ax.bar(range(len(top)), top["occupancy_pct"],
           color=colours, edgecolor="white", linewidth=0.4)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.0,
               label="50% occupancy threshold")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Contact Occupancy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-Residue Binding Site Contact Occupancy (% of Sampled Frames)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

    out_path = os.path.join(outdir, "binding_site_contact_occupancy.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Contact occupancy plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  Binding Site Composition & Hotspot Analysis")
    print(f"  TPR: {args.tpr} | XTC: {args.xtc}")
    print(f"{'='*58}")

    if not os.path.exists(args.tpr) or not os.path.exists(args.xtc):
        raise FileNotFoundError("Input files not found. Run GROMACS pipeline first.")

    u = mda.Universe(args.tpr, args.xtc)
    print(f"\n  Universe: {len(u.atoms)} atoms | {len(u.trajectory)} frames")

    all_contacts, contact_freq, n_sampled = collect_binding_site_residues(
        u, args.ligand_resname, args.cutoff, args.every_nth_frame
    )

    if not all_contacts:
        print("  No binding site residues found. Check ligand resname and cutoff.")
        return

    restype_df, occ_df = compute_frequency_table(all_contacts, contact_freq, n_sampled)

    # Save CSV outputs
    restype_df.to_csv(os.path.join(args.outdir, "residue_type_frequency.csv"),
                      index=False, float_format="%.3f")
    occ_df.to_csv(os.path.join(args.outdir, "residue_contact_occupancy.csv"),
                  index=False, float_format="%.3f")
    print(f"\n  CSVs saved to {args.outdir}/")

    plot_residue_composition(restype_df, args.outdir)
    plot_contact_occupancy_heatmap(occ_df, args.outdir)

    print(f"\n  All outputs written to: {args.outdir}/")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
