#!/usr/bin/env python3
"""
analysis_hbond.py

Per-residue protein-ligand hydrogen bond occupancy over an MD trajectory.
This is a Python/MDAnalysis equivalent of the existence-matrix analysis
originally run with GROMACS `gmx bond` (-hbn/-hbm flags).

Occupancy is reported both as a continuous percentage and binned into the
three time windows used in the original analysis:
  low      0-30 ns   (first 30% of a 100 ns run)
  moderate 30-70 ns
  high     70-100 ns

Author: Linta Mahboob
Requires: MDAnalysis >= 2.4, numpy, matplotlib, pandas
Usage:
    python analysis_hbond.py --tpr md.tpr --xtc md_noPBC.xtc --ligand-resname LIG
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required: pip install pandas")

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
except ImportError:
    raise ImportError("MDAnalysis is required: pip install MDAnalysis")

COLOR_DEFAULT = "#5B85AA"
COLOR_HIGH = "#1E7C4B"
NON_PROTEIN_RESNAMES = {"HOH", "WAT", "SOL", "NA", "CL", "K", "MG", "CA", "ZN"}


def parse_args():
    p = argparse.ArgumentParser(description="Protein-ligand hydrogen bond occupancy analysis")
    p.add_argument("--tpr", default="md.tpr")
    p.add_argument("--xtc", default="md_noPBC.xtc")
    p.add_argument("--ligand-resname", default="LIG")
    p.add_argument("--outdir", default="analysis_output")
    p.add_argument("--distance", type=float, default=3.5, help="Donor-acceptor distance cutoff (A)")
    p.add_argument("--angle", type=float, default=150.0, help="D-H-A angle cutoff (deg)")
    return p.parse_args()


def run_hbond_analysis(universe, ligand_resname, d_cutoff, a_cutoff):
    hbonds = HBA(
        universe=universe,
        between=["protein", f"resname {ligand_resname}"],
        d_a_cutoff=d_cutoff,
        d_h_a_angle_cutoff=a_cutoff,
        update_selections=False,
    )
    hbonds.run(verbose=True)
    return hbonds


def compute_occupancy(hbonds, n_frames, ligand_resname):
    """Fraction of frames in which each protein residue forms an H-bond with the ligand."""
    results = hbonds.results.hbonds
    if len(results) == 0:
        return pd.DataFrame()

    rows = [(int(r[0]), int(r[1]), int(r[3])) for r in results]
    df = pd.DataFrame(rows, columns=["frame", "donor_resid", "acceptor_resid"])

    resid_to_resname = {a.resid: a.resname for a in hbonds._universe.atoms}
    df["donor_resname"] = df["donor_resid"].map(resid_to_resname)
    df["acceptor_resname"] = df["acceptor_resid"].map(resid_to_resname)

    def is_protein(resname):
        return resname != ligand_resname.upper() and resname not in NON_PROTEIN_RESNAMES

    occ_data = []
    for _, row in df.iterrows():
        if is_protein(row["donor_resname"]):
            occ_data.append((row["frame"], row["donor_resid"], row["donor_resname"]))
        elif is_protein(row["acceptor_resname"]):
            occ_data.append((row["frame"], row["acceptor_resid"], row["acceptor_resname"]))

    if not occ_data:
        return pd.DataFrame()

    occ_df = pd.DataFrame(occ_data, columns=["frame", "resid", "resname"])
    occupancy = (
        occ_df.groupby(["resid", "resname"])["frame"].nunique()
        .reset_index().rename(columns={"frame": "frames_with_hbond"})
    )
    occupancy["occupancy_frac"] = occupancy["frames_with_hbond"] / n_frames
    occupancy["occupancy_percent"] = occupancy["occupancy_frac"] * 100
    occupancy = occupancy.sort_values("occupancy_frac", ascending=False).reset_index(drop=True)
    return occupancy


def assign_time_cluster(occupancy, low_thresh=0.30, high_thresh=0.70):
    """
    Bin residues into low/moderate/high occupancy clusters using the same
    proportional windows as the original gmx bond existence-matrix analysis
    (0-30% / 30-70% / 70-100% of total simulation time).
    """
    bins = [-np.inf, low_thresh, high_thresh, np.inf]
    labels = ["low", "moderate", "high"]
    occupancy["cluster"] = pd.cut(occupancy["occupancy_frac"], bins=bins, labels=labels)
    return occupancy


def plot_hbond_count(hbonds, n_frames, outdir):
    results = hbonds.results.hbonds
    if len(results) == 0:
        return
    frames = results[:, 0].astype(int)
    n_bonds = np.bincount(frames, minlength=n_frames)
    times = np.arange(n_frames) * hbonds._universe.trajectory.dt / 1000.0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(times, n_bonds, alpha=0.3, color=COLOR_DEFAULT)
    ax.plot(times, n_bonds, color=COLOR_DEFAULT, linewidth=0.8)
    ax.axhline(np.mean(n_bonds), color="gray", linestyle="--", linewidth=1.0,
               label=f"Mean {np.mean(n_bonds):.1f}")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("H-bond count")
    ax.set_title("Protein-ligand hydrogen bond count over time")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(outdir, "hbond_count_vs_time.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_occupancy_bar(occupancy, outdir):
    if occupancy.empty:
        return
    top = occupancy.head(30).copy()
    colors = [COLOR_HIGH if f >= 0.70 else COLOR_DEFAULT for f in top["occupancy_frac"]]
    labels = top.apply(lambda r: f"{r['resname']}{r['resid']}", axis=1)

    fig, ax = plt.subplots(figsize=(max(9, len(top) * 0.45), 4.5))
    ax.bar(range(len(top)), top["occupancy_percent"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("H-bond occupancy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-residue hydrogen bond occupancy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(outdir, "hbond_occupancy_per_residue.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def save_occupancy_csv(occupancy, outdir):
    if occupancy.empty:
        return
    path = os.path.join(outdir, "hbond_occupancy.csv")
    occupancy.to_csv(path, index=False, float_format="%.4f")
    print(f"Saved {path}")

    counts = occupancy["cluster"].value_counts()
    total = len(occupancy)
    print("\nOccupancy cluster summary")
    for label in ["low", "moderate", "high"]:
        n = counts.get(label, 0)
        print(f"  {label:<9} {n:>4}  ({n / total * 100:.1f}%)")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.tpr) or not os.path.exists(args.xtc):
        raise FileNotFoundError(f"Missing input files: {args.tpr}, {args.xtc}")

    u = mda.Universe(args.tpr, args.xtc)
    n_frames = len(u.trajectory)
    print(f"Loaded {len(u.atoms)} atoms, {n_frames} frames")

    hbonds = run_hbond_analysis(u, args.ligand_resname, args.distance, args.angle)
    occupancy = compute_occupancy(hbonds, n_frames, args.ligand_resname)
    occupancy = assign_time_cluster(occupancy)

    plot_hbond_count(hbonds, n_frames, args.outdir)
    plot_occupancy_bar(occupancy, args.outdir)
    save_occupancy_csv(occupancy, args.outdir)

    print(f"\nAll outputs written to {args.outdir}/")


if __name__ == "__main__":
    main()
