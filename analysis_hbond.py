#!/usr/bin/env python3
"""
analysis_hbond.py
─────────────────
Analyses protein-ligand hydrogen bond occupancy over a 100 ns MD trajectory.

Key output metrics (consistent with thesis findings):
  • Per-residue H-bond occupancy (% of simulation frames)
  • High-occupancy residues (> 50% occupancy threshold, as per thesis: 86.5%)
  • H-bond count vs time plot
  • Occupancy bar chart coloured by residue type (Asp / His / Arg highlighted)

Author : Linta Mahboob | University of Siena
Requires: MDAnalysis >= 2.4, numpy, matplotlib, pandas
Usage  : python analysis_hbond.py [--tpr md.tpr] [--xtc md_noPBC.xtc]
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas required: pip install pandas")

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
        HydrogenBondAnalysis as HBA,
    )
except ImportError:
    raise ImportError("MDAnalysis required: pip install MDAnalysis")

warnings.filterwarnings("ignore")

# ─── Colour palette ───────────────────────────────────────────────────────────
COLOUR_DEFAULT = "#5B85AA"   # default bar colour
COLOUR_ASP     = "#C0392B"   # Aspartate  — thesis hotspot #1
COLOUR_HIS     = "#8E44AD"   # Histidine  — thesis hotspot #2
COLOUR_ARG     = "#D35400"   # Arginine   — thesis hotspot #3
COLOUR_HIGH    = "#1E7C4B"   # Other high-occupancy residue

HOTSPOT_RESIDUES = {"ASP": COLOUR_ASP, "HIS": COLOUR_HIS, "ARG": COLOUR_ARG}
HIGH_OCC_THRESH  = 0.50   # 50% occupancy = "high occupancy" (thesis: 86.5% of residues pass)


def parse_args():
    p = argparse.ArgumentParser(
        description="Protein-ligand hydrogen bond occupancy analysis"
    )
    p.add_argument("--tpr",  default="md.tpr",        help="GROMACS .tpr file")
    p.add_argument("--xtc",  default="md_noPBC.xtc",  help="PBC-corrected trajectory")
    p.add_argument("--ligand-resname", default="LIG",  help="Ligand residue name")
    p.add_argument("--outdir", default="analysis_output",
                   help="Output directory (default: analysis_output)")
    p.add_argument("--distance",  type=float, default=3.5,
                   help="H-bond donor-acceptor distance cutoff Å (default: 3.5)")
    p.add_argument("--angle",     type=float, default=150.0,
                   help="H-bond angle cutoff degrees (default: 150)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def run_hbond_analysis(universe, ligand_resname, d_cutoff, a_cutoff):
    """
    Run full protein–ligand H-bond analysis using MDAnalysis HBA.
    Returns the HBA object after running on the full trajectory.
    """
    print(f"\n  Running H-bond analysis (d ≤ {d_cutoff} Å, angle ≥ {a_cutoff}°) ...")

    hbonds = HBA(
        universe=universe,
        donors_sel=None,       # auto-detect from topology
        hydrogens_sel=None,    # auto-detect
        acceptors_sel=None,    # auto-detect
        between=["protein", f"resname {ligand_resname}"],
        d_a_cutoff=d_cutoff,
        d_h_a_angle_cutoff=a_cutoff,
        update_selections=False,
    )
    hbonds.run(verbose=True)
    return hbonds


# ─────────────────────────────────────────────────────────────────────────────
def compute_occupancy(hbonds, n_frames, ligand_resname):
    """
    Compute per-residue H-bond occupancy (fraction of frames with ≥1 H-bond).
    Returns a DataFrame sorted by descending occupancy.
    """
    results = hbonds.results.hbonds   # shape: (N_hbonds, 8)

    if len(results) == 0:
        print("  Warning: No hydrogen bonds detected. Check selection / cutoffs.")
        return pd.DataFrame()

    # Column order: frame, donor_idx, hydrogen_idx, acceptor_idx,
    #               distance, angle, donor_resnm, acceptor_resnm
    # We classify based on which atom belongs to the protein
    rows = []
    for row in results:
        frame       = int(row[0])
        don_resid   = int(row[1])    # donor resid in universe
        acc_resid   = int(row[3])    # acceptor resid in universe
        dist        = float(row[4])
        angle       = float(row[5])
        rows.append((frame, don_resid, acc_resid, dist, angle))

    df = pd.DataFrame(rows, columns=["frame", "donor_resid", "acceptor_resid",
                                     "distance_A", "angle_deg"])

    # Map resid → resname using universe atom group
    universe = hbonds._universe
    resid_to_resname = {
        atom.resid: atom.resname for atom in universe.atoms
    }
    df["donor_resname"]    = df["donor_resid"].map(resid_to_resname)
    df["acceptor_resname"] = df["acceptor_resid"].map(resid_to_resname)

    # Identify protein residues (exclude ligand)
    def is_protein_res(resname):
        return resname != ligand_resname.upper() and resname not in (
            "HOH", "WAT", "SOL", "NA", "CL", "K", "MG", "CA", "ZN"
        )

    # Determine which residue is the protein partner
    occ_data = []
    for _, row in df.iterrows():
        if is_protein_res(row["donor_resname"]):
            occ_data.append((row["frame"], row["donor_resid"],
                             row["donor_resname"]))
        elif is_protein_res(row["acceptor_resname"]):
            occ_data.append((row["frame"], row["acceptor_resid"],
                             row["acceptor_resname"]))

    if not occ_data:
        return pd.DataFrame()

    occ_df = pd.DataFrame(occ_data, columns=["frame", "resid", "resname"])

    # Count unique frames per residue
    occupancy = (
        occ_df.groupby(["resid", "resname"])["frame"]
        .nunique()
        .reset_index()
        .rename(columns={"frame": "frames_with_hbond"})
    )
    occupancy["occupancy_frac"]    = occupancy["frames_with_hbond"] / n_frames
    occupancy["occupancy_percent"] = occupancy["occupancy_frac"] * 100
    occupancy = occupancy.sort_values("occupancy_frac", ascending=False)

    high_occ = (occupancy["occupancy_frac"] >= HIGH_OCC_THRESH).sum()
    print(f"\n  Total unique protein residues forming H-bonds: {len(occupancy)}")
    print(f"  High-occupancy residues (≥{HIGH_OCC_THRESH*100:.0f}%): {high_occ} "
          f"({high_occ/len(occupancy)*100:.1f}%)")
    print(f"  [Thesis benchmark: 86.5% of H-bonding residues show high occupancy]")

    return occupancy


# ─────────────────────────────────────────────────────────────────────────────
def plot_hbond_count(hbonds, n_frames, outdir):
    """Plot number of H-bonds vs simulation time."""
    results = hbonds.results.hbonds
    if len(results) == 0:
        return

    # Count H-bonds per frame
    frames  = results[:, 0].astype(int)
    n_bonds = np.bincount(frames, minlength=n_frames)
    times   = np.arange(n_frames) * hbonds._universe.trajectory.dt / 1000.0

    # Running average (window = 5% of trajectory)
    window = max(1, int(n_frames * 0.05))
    running_avg = np.convolve(n_bonds, np.ones(window)/window, mode="same")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(times, n_bonds, alpha=0.3, color=COLOUR_DEFAULT,
                    label="H-bond count")
    ax.plot(times, running_avg, color=COLOUR_DEFAULT, linewidth=1.5,
            label=f"Running avg (window={window} frames)")
    ax.axhline(np.mean(n_bonds), color="gray", linestyle="--",
               linewidth=1.2, label=f"Mean = {np.mean(n_bonds):.1f}")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Number of H-bonds")
    ax.set_title("Protein–Ligand Hydrogen Bond Count vs Time (100 ns)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(outdir, "hbond_count_vs_time.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  H-bond count plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_occupancy_bar(occupancy, outdir):
    """
    Bar chart of per-residue H-bond occupancy.
    Bars coloured by residue type: Asp (red), His (purple), Arg (orange),
    other high-occ (green), rest (blue).
    """
    if occupancy.empty:
        return

    # Limit to top 30 for readability
    top = occupancy.head(30).copy()

    # Assign colours
    def get_colour(row):
        rn = row["resname"].upper()
        if rn in HOTSPOT_RESIDUES:
            return HOTSPOT_RESIDUES[rn]
        if row["occupancy_frac"] >= HIGH_OCC_THRESH:
            return COLOUR_HIGH
        return COLOUR_DEFAULT

    top["colour"] = top.apply(get_colour, axis=1)
    labels = top.apply(lambda r: f"{r['resname']}{r['resid']}", axis=1)

    fig, ax = plt.subplots(figsize=(max(10, len(top)*0.5), 5))
    bars = ax.bar(range(len(top)), top["occupancy_percent"],
                  color=top["colour"], edgecolor="white", linewidth=0.5)

    ax.axhline(HIGH_OCC_THRESH * 100, color="gray", linestyle="--",
               linewidth=1.2, label=f"High-occupancy threshold ({HIGH_OCC_THRESH*100:.0f}%)")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("H-bond Occupancy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-Residue Hydrogen Bond Occupancy — Protein–Ligand Contacts")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLOUR_ASP,     label="Aspartate (Asp) — Dynamic Hotspot"),
        mpatches.Patch(color=COLOUR_HIS,     label="Histidine (His) — Dynamic Hotspot"),
        mpatches.Patch(color=COLOUR_ARG,     label="Arginine (Arg)  — Dynamic Hotspot"),
        mpatches.Patch(color=COLOUR_HIGH,    label="Other high-occupancy residue"),
        mpatches.Patch(color=COLOUR_DEFAULT, label="Low-occupancy residue"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")

    out_path = os.path.join(outdir, "hbond_occupancy_per_residue.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Occupancy bar chart saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def save_occupancy_csv(occupancy, outdir):
    if occupancy.empty:
        return
    csv_path = os.path.join(outdir, "hbond_occupancy.csv")
    occupancy.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Occupancy data saved → {csv_path}")

    # Print top 10
    print("\n  Top 10 H-bonding residues by occupancy:")
    print(f"  {'Residue':<12} {'Resname':<8} {'Occupancy':>12}  {'High Occ?':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*12}  {'-'*10}")
    for _, row in occupancy.head(10).iterrows():
        is_high = "YES" if row["occupancy_frac"] >= HIGH_OCC_THRESH else "no"
        print(f"  {row['resname']:<12} {row['resid']:<8} "
              f"{row['occupancy_percent']:>10.1f}%  {is_high:>10}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  Hydrogen Bond Occupancy Analysis")
    print(f"  TPR : {args.tpr} | XTC : {args.xtc}")
    print(f"  Ligand: {args.ligand_resname}")
    print(f"{'='*58}")

    if not os.path.exists(args.tpr) or not os.path.exists(args.xtc):
        raise FileNotFoundError(
            f"Input files not found. Run GROMACS pipeline first."
        )

    u = mda.Universe(args.tpr, args.xtc)
    n_frames = len(u.trajectory)
    print(f"\n  Universe: {len(u.atoms)} atoms | {n_frames} frames")

    hbonds   = run_hbond_analysis(u, args.ligand_resname,
                                  args.distance, args.angle)
    occupancy = compute_occupancy(hbonds, n_frames, args.ligand_resname)

    plot_hbond_count(hbonds, n_frames, args.outdir)
    plot_occupancy_bar(occupancy, args.outdir)
    save_occupancy_csv(occupancy, args.outdir)

    print(f"\n  All H-bond outputs written to: {args.outdir}/")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
