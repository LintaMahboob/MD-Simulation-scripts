#!/usr/bin/env python3
"""
analysis_rmsd_sasa.py

Computes RMSD (protein Ca, binding-residue backbone, ligand) and binding-site
SASA from a GROMACS MD trajectory, and writes plots + summary statistics.

Author: Linta Mahboob
Requires: MDAnalysis >= 2.4, numpy, matplotlib, scipy
Usage:
    python analysis_rmsd_sasa.py --tpr md.tpr --xtc md_noPBC.xtc --ligand-resname LIG
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
except ImportError:
    raise ImportError("MDAnalysis is required: pip install MDAnalysis")

try:
    from MDAnalysis.analysis import sasa as mda_sasa
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_PROTEIN = "#2C5F8A"
COLOR_LIGAND = "#C0392B"
COLOR_SASA = "#1E7C4B"


def parse_args():
    p = argparse.ArgumentParser(description="RMSD and SASA analysis for a protein-ligand MD trajectory")
    p.add_argument("--tpr", default="md.tpr", help="GROMACS run input file")
    p.add_argument("--xtc", default="md_noPBC.xtc", help="PBC-corrected trajectory")
    p.add_argument("--ligand-resname", default="LIG")
    p.add_argument("--binding-residues", default=None,
                    help="Comma-separated resids for binding-site SASA; auto-detected if omitted")
    p.add_argument("--cutoff", type=float, default=5.0,
                    help="Distance cutoff (A) for auto-detecting binding-site residues")
    p.add_argument("--outdir", default="analysis_output")
    return p.parse_args()


def compute_rmsd(universe, ligand_resname):
    """
    RMSD of protein Ca atoms and ligand heavy atoms relative to the first frame,
    after alignment on the protein Ca backbone.
    """
    ref = mda.Universe(universe.filename, universe.trajectory.filename)
    align.AlignTraj(universe, ref, select="protein and name CA", in_memory=False).run()

    sel_ca = universe.select_atoms("protein and name CA")
    sel_lig = universe.select_atoms(f"resname {ligand_resname} and not name H*")

    universe.trajectory[0]
    ref_ca = sel_ca.positions.copy()
    ref_lig = sel_lig.positions.copy() if len(sel_lig) > 0 else None

    times, rmsd_ca, rmsd_lig = [], [], []
    for ts in universe.trajectory:
        times.append(ts.time / 1000.0)  # ps -> ns
        rmsd_ca.append(np.sqrt(np.mean(np.sum((sel_ca.positions - ref_ca) ** 2, axis=1))))
        if ref_lig is not None:
            rmsd_lig.append(np.sqrt(np.mean(np.sum((sel_lig.positions - ref_lig) ** 2, axis=1))))
        else:
            rmsd_lig.append(np.nan)

    return np.array(times), np.array(rmsd_ca), np.array(rmsd_lig)


def compute_sasa(universe, binding_residues, ligand_resname, cutoff):
    """
    Per-frame SASA of the binding-site residue backbone.
    Binding-site residues are auto-detected (within `cutoff` A of the ligand
    in the first frame) unless explicitly provided.
    """
    if binding_residues is None:
        universe.trajectory[0]
        lig_atoms = universe.select_atoms(f"resname {ligand_resname}")
        if len(lig_atoms) == 0:
            print(f"Ligand '{ligand_resname}' not found; skipping SASA.")
            return None, None
        shell = universe.select_atoms(f"protein and around {cutoff} resname {ligand_resname}")
        binding_residues = sorted(set(shell.resids))

    res_sel = "resid " + " ".join(str(r) for r in binding_residues)
    binding_atoms = universe.select_atoms(f"protein and ({res_sel})")

    if not HAS_FREESASA:
        print("freesasa not available; install with: pip install freesasa")
        return None, None

    times, sasa_values = [], []
    for ts in universe.trajectory:
        times.append(ts.time / 1000.0)
        result = mda_sasa.SASA(binding_atoms).run()
        sasa_values.append(result.results.total_area[0])

    return np.array(times), np.array(sasa_values)


def plot_rmsd(times, rmsd_ca, rmsd_lig, outdir):
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, rmsd_ca, color=COLOR_PROTEIN, linewidth=0.7)
    ax1.axhline(np.median(rmsd_ca), color=COLOR_PROTEIN, linestyle="--",
                linewidth=1.2, label=f"Median {np.median(rmsd_ca):.2f} A")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("RMSD (A)")
    ax1.set_title("Protein Ca RMSD")
    ax1.legend(fontsize=8)

    mask = ~np.isnan(rmsd_lig)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times[mask], rmsd_lig[mask], color=COLOR_LIGAND, linewidth=0.7)
    lig_med = np.nanmedian(rmsd_lig)
    ax2.axhline(lig_med, color=COLOR_LIGAND, linestyle="--", linewidth=1.2,
                label=f"Median {lig_med:.2f} A")
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("RMSD (A)")
    ax2.set_title("Ligand RMSD")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(rmsd_ca, bins=40, color=COLOR_PROTEIN, alpha=0.75, edgecolor="white")
    ax3.set_xlabel("RMSD (A)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Protein RMSD distribution")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(rmsd_lig[mask], bins=40, color=COLOR_LIGAND, alpha=0.75, edgecolor="white")
    ax4.set_xlabel("RMSD (A)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Ligand RMSD distribution")

    fig.suptitle("RMSD Analysis - Protein-Ligand MD Trajectory", fontsize=12)
    out_path = os.path.join(outdir, "rmsd_analysis.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_sasa(times, sasa_values, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(times, sasa_values, color=COLOR_SASA, linewidth=0.7)
    med = np.median(sasa_values)
    ax1.axhline(med, color=COLOR_SASA, linestyle="--", linewidth=1.2,
                label=f"Median {med:.2f} A2")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("SASA (A2)")
    ax1.set_title("Binding-site SASA")
    ax1.legend(fontsize=8)

    ax2.hist(sasa_values, bins=40, color=COLOR_SASA, alpha=0.75, edgecolor="white")
    ax2.set_xlabel("SASA (A2)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("SASA distribution")

    fig.suptitle("Binding-Site SASA - Protein-Ligand MD Trajectory", fontsize=12)
    out_path = os.path.join(outdir, "sasa_analysis.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def save_summary(times, rmsd_ca, rmsd_lig, sasa_values, outdir):
    mask = ~np.isnan(rmsd_lig)
    path = os.path.join(outdir, "summary_statistics.txt")
    with open(path, "w") as f:
        f.write("MD simulation summary\n\n")
        f.write("Protein Ca RMSD (A)\n")
        f.write(f"  mean {rmsd_ca.mean():.3f}  median {np.median(rmsd_ca):.3f}  "
                f"sd {rmsd_ca.std():.3f}  min {rmsd_ca.min():.3f}  max {rmsd_ca.max():.3f}\n\n")
        f.write("Ligand RMSD (A)\n")
        f.write(f"  mean {np.nanmean(rmsd_lig):.3f}  median {np.nanmedian(rmsd_lig):.3f}  "
                f"sd {np.nanstd(rmsd_lig):.3f}  min {np.nanmin(rmsd_lig):.3f}  max {np.nanmax(rmsd_lig):.3f}\n\n")
        if sasa_values is not None:
            f.write("Binding-site SASA (A2)\n")
            f.write(f"  mean {sasa_values.mean():.3f}  median {np.median(sasa_values):.3f}  "
                    f"sd {sasa_values.std():.3f}  min {sasa_values.min():.3f}  max {sasa_values.max():.3f}\n\n")
        f.write("Simulation parameters\n")
        f.write(f"  total time  : {times[-1]:.1f} ns\n")
        f.write(f"  frames      : {len(times)}\n")
        f.write( "  force field : CHARMM36-mar2019\n")
        f.write( "  water model : TIP3P\n")
        f.write( "  thermostat  : V-rescale, 300 K\n")
        f.write( "  barostat    : Nose-Hoover, 1 atm\n")
    print(f"Saved {path}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.tpr) or not os.path.exists(args.xtc):
        raise FileNotFoundError(f"Missing input files: {args.tpr}, {args.xtc}")

    u = mda.Universe(args.tpr, args.xtc)
    print(f"Loaded {len(u.atoms)} atoms, {len(u.trajectory)} frames")

    binding_res = None
    if args.binding_residues:
        binding_res = [int(r.strip()) for r in args.binding_residues.split(",")]

    times, rmsd_ca, rmsd_lig = compute_rmsd(u, args.ligand_resname)
    _, sasa_values = compute_sasa(u, binding_res, args.ligand_resname, args.cutoff)

    np.savetxt(os.path.join(args.outdir, "rmsd_ca.dat"),
               np.column_stack([times, rmsd_ca]), header="time_ns rmsd_ca_A", fmt="%.4f")
    np.savetxt(os.path.join(args.outdir, "rmsd_ligand.dat"),
               np.column_stack([times, rmsd_lig]), header="time_ns rmsd_lig_A", fmt="%.4f")
    if sasa_values is not None:
        np.savetxt(os.path.join(args.outdir, "sasa_binding_site.dat"),
                   np.column_stack([times, sasa_values]), header="time_ns sasa_A2", fmt="%.4f")

    plot_rmsd(times, rmsd_ca, rmsd_lig, args.outdir)
    if sasa_values is not None:
        plot_sasa(times, sasa_values, args.outdir)

    save_summary(times, rmsd_ca, rmsd_lig, sasa_values, args.outdir)
    print(f"All outputs written to {args.outdir}/")


if __name__ == "__main__":
    main()
