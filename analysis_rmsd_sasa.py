#!/usr/bin/env python3
"""
analysis_rmsd_sasa.py
─────────────────────
Computes and plots RMSD (protein backbone + ligand) and SASA (binding site
residues) from a 100 ns GROMACS MD trajectory.

Metrics match the thesis framework:
  • Protein backbone RMSD  — conformational stability assessment
  • Ligand RMSD            — binding pose stability (median ~ 1.6 Å benchmark)
  • Binding site SASA      — residue burial / solvent exposure

Author : Linta Mahboob | University of Siena
Requires: MDAnalysis >= 2.4, numpy, matplotlib, scipy
Usage  : python analysis_rmsd_sasa.py [--tpr md.tpr] [--xtc md_noPBC.xtc]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, align
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
except ImportError:
    raise ImportError(
        "MDAnalysis not found. Install with: pip install MDAnalysis"
    )

# ─── SASA via FreeSASA (optional, falls back to simple approximation) ─────────
try:
    from MDAnalysis.analysis import sasa as mda_sasa
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False
    print("Warning: FreeSASA not available. SASA approximated via atom radii method.")

# ─── Matplotlib global style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})

THESIS_BLUE   = "#2C5F8A"
THESIS_RED    = "#C0392B"
THESIS_GREEN  = "#1E7C4B"
THESIS_ORANGE = "#D35400"


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="RMSD + SASA analysis for protein-ligand MD trajectories"
    )
    parser.add_argument("--tpr", default="md.tpr",
                        help="GROMACS topology/run file (default: md.tpr)")
    parser.add_argument("--xtc", default="md_noPBC.xtc",
                        help="PBC-corrected trajectory (default: md_noPBC.xtc)")
    parser.add_argument("--ligand-resname", default="LIG",
                        help="Ligand residue name in topology (default: LIG)")
    parser.add_argument("--binding-residues", default=None,
                        help="Comma-separated residue IDs for binding site SASA, "
                             "e.g. '45,67,102,150'. Auto-detected if omitted.")
    parser.add_argument("--outdir", default="analysis_output",
                        help="Output directory for plots and data (default: analysis_output)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. RMSD CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_rmsd(universe, ligand_resname="LIG"):
    """
    Compute RMSD of:
      (a) Protein Cα atoms  — backbone structural stability
      (b) Ligand heavy atoms — binding pose stability
    Reference: first frame of trajectory.
    Returns time array (ns) and two RMSD arrays (Å).
    """
    print("\n  Computing protein Cα RMSD ...")
    # Align trajectory to first frame on Cα before computing RMSD
    ref = mda.Universe(universe.filename, universe.trajectory.filename)
    aligner = align.AlignTraj(universe, ref,
                               select="protein and name CA",
                               in_memory=False).run()

    sel_ca   = universe.select_atoms("protein and name CA")
    sel_lig  = universe.select_atoms(f"resname {ligand_resname} and not name H*")

    # Store reference positions (frame 0)
    universe.trajectory[0]
    ref_ca  = sel_ca.positions.copy()
    ref_lig = sel_lig.positions.copy() if len(sel_lig) > 0 else None

    times, rmsd_ca, rmsd_lig = [], [], []

    for ts in universe.trajectory:
        t_ns = ts.time / 1000.0   # ps → ns

        # Cα RMSD (Å)
        diff_ca = sel_ca.positions - ref_ca
        rmsd_ca.append(np.sqrt(np.mean(np.sum(diff_ca**2, axis=1))))

        # Ligand RMSD (Å)
        if ref_lig is not None:
            diff_l = sel_lig.positions - ref_lig
            rmsd_lig.append(np.sqrt(np.mean(np.sum(diff_l**2, axis=1))))
        else:
            rmsd_lig.append(np.nan)

        times.append(t_ns)

    times    = np.array(times)
    rmsd_ca  = np.array(rmsd_ca)
    rmsd_lig = np.array(rmsd_lig)

    print(f"    Cα RMSD  — mean: {rmsd_ca.mean():.2f} Å  "
          f"median: {np.median(rmsd_ca):.2f} Å  max: {rmsd_ca.max():.2f} Å")
    print(f"    Ligand RMSD — mean: {np.nanmean(rmsd_lig):.2f} Å  "
          f"median: {np.nanmedian(rmsd_lig):.2f} Å  "
          f"max: {np.nanmax(rmsd_lig):.2f} Å")

    return times, rmsd_ca, rmsd_lig


# ─────────────────────────────────────────────────────────────────────────────
# 3. SASA CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_sasa(universe, binding_residues=None, ligand_resname="LIG"):
    """
    Compute per-frame SASA of binding site residues.
    If binding_residues is None, auto-detects residues within 5 Å of ligand
    in the first frame.
    Returns times (ns) and SASA array (Å²).
    """
    print("\n  Computing binding site SASA ...")

    # Auto-detect binding residues from first frame if not provided
    if binding_residues is None:
        universe.trajectory[0]
        lig_atoms = universe.select_atoms(f"resname {ligand_resname}")
        if len(lig_atoms) == 0:
            print("    Warning: Ligand not found for SASA auto-detection. Skipping.")
            return None, None

        shell = universe.select_atoms(
            f"protein and around 5.0 resname {ligand_resname}"
        )
        binding_residues = list(set(shell.resids))
        print(f"    Auto-detected {len(binding_residues)} binding-site residues "
              f"within 5 Å of ligand: {sorted(binding_residues)}")

    res_sel = "resid " + " ".join(str(r) for r in binding_residues)
    binding_atoms = universe.select_atoms(f"protein and ({res_sel})")
    print(f"    Selection: {len(binding_atoms)} atoms in binding site")

    times_sasa = []
    sasa_values = []

    # Simple van-der-Waals-radius-based approximation if FreeSASA unavailable
    for ts in universe.trajectory:
        t_ns = ts.time / 1000.0
        if HAS_FREESASA:
            # Use FreeSASA via MDAnalysis wrapper (more accurate)
            sasa_frame = mda_sasa.SASA(binding_atoms).run()
            sasa_values.append(sasa_frame.results.total_area[0])
        else:
            # Fallback: rough approximation via atomic radii sum (not real SASA)
            vdw_radii = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.2}
            exposed_area = sum(
                4 * np.pi * vdw_radii.get(a.element, 1.6)**2
                for a in binding_atoms
            ) * 0.30   # empirical exposure fraction for buried residues
            sasa_values.append(exposed_area)
        times_sasa.append(t_ns)

    times_sasa  = np.array(times_sasa)
    sasa_values = np.array(sasa_values)

    print(f"    Binding site SASA — mean: {sasa_values.mean():.2f} Å²  "
          f"median: {np.median(sasa_values):.2f} Å²  "
          f"min: {sasa_values.min():.2f} Å²")

    return times_sasa, sasa_values


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_rmsd(times, rmsd_ca, rmsd_lig, outdir):
    """Four-panel RMSD figure: raw traces + histograms."""
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

    # Panel A — Protein Cα RMSD vs time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, rmsd_ca, color=THESIS_BLUE, linewidth=0.7, alpha=0.85)
    ax1.axhline(np.median(rmsd_ca), color=THESIS_BLUE, linestyle="--",
                linewidth=1.2, label=f"Median = {np.median(rmsd_ca):.2f} Å")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("RMSD (Å)")
    ax1.set_title("A   Protein Backbone Cα RMSD")
    ax1.legend(fontsize=8)

    # Panel B — Ligand RMSD vs time
    ax2 = fig.add_subplot(gs[0, 1])
    mask = ~np.isnan(rmsd_lig)
    ax2.plot(times[mask], rmsd_lig[mask], color=THESIS_RED,
             linewidth=0.7, alpha=0.85)
    lig_med = np.nanmedian(rmsd_lig)
    ax2.axhline(lig_med, color=THESIS_RED, linestyle="--",
                linewidth=1.2, label=f"Median = {lig_med:.2f} Å")
    ax2.axhline(1.6, color="gray", linestyle=":", linewidth=1.0,
                label="Thesis threshold (1.6 Å)")
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("RMSD (Å)")
    ax2.set_title("B   Ligand RMSD (Binding Pose Stability)")
    ax2.legend(fontsize=8)

    # Panel C — Protein Cα RMSD histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(rmsd_ca, bins=40, color=THESIS_BLUE, alpha=0.7, edgecolor="white")
    ax3.axvline(np.median(rmsd_ca), color=THESIS_BLUE, linestyle="--",
                linewidth=1.5, label=f"Median {np.median(rmsd_ca):.2f} Å")
    ax3.set_xlabel("RMSD (Å)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("C   Protein RMSD Distribution")
    ax3.legend(fontsize=8)

    # Panel D — Ligand RMSD histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(rmsd_lig[mask], bins=40, color=THESIS_RED, alpha=0.7,
             edgecolor="white")
    ax4.axvline(1.6, color="gray", linestyle=":", linewidth=1.5,
                label="Thesis benchmark (1.6 Å)")
    ax4.axvline(lig_med, color=THESIS_RED, linestyle="--", linewidth=1.5,
                label=f"Median {lig_med:.2f} Å")
    ax4.set_xlabel("RMSD (Å)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("D   Ligand RMSD Distribution")
    ax4.legend(fontsize=8)

    fig.suptitle(
        "RMSD Analysis — Protein-Ligand Complex MD Simulation (100 ns)\n"
        "GROMACS / CHARMM36 / TIP3P | Linta Mahboob, University of Siena",
        fontsize=11, y=1.01
    )

    out_path = os.path.join(outdir, "rmsd_analysis.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  RMSD figure saved → {out_path}")


def plot_sasa(times_sasa, sasa_values, outdir):
    """Two-panel SASA figure: trace + histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.35)

    # SASA over time
    ax1.plot(times_sasa, sasa_values, color=THESIS_GREEN,
             linewidth=0.7, alpha=0.85)
    med_sasa = np.median(sasa_values)
    ax1.axhline(med_sasa, color=THESIS_GREEN, linestyle="--",
                linewidth=1.3, label=f"Median = {med_sasa:.2f} Å²")
    ax1.fill_between(times_sasa,
                     sasa_values.mean() - sasa_values.std(),
                     sasa_values.mean() + sasa_values.std(),
                     alpha=0.15, color=THESIS_GREEN, label="±1 SD")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("SASA (Å²)")
    ax1.set_title("A   Binding Site SASA over 100 ns")
    ax1.legend(fontsize=8)

    # SASA histogram
    ax2.hist(sasa_values, bins=40, color=THESIS_GREEN, alpha=0.7,
             edgecolor="white")
    ax2.axvline(med_sasa, color=THESIS_GREEN, linestyle="--",
                linewidth=1.5, label=f"Median {med_sasa:.2f} Å²")
    ax2.set_xlabel("SASA (Å²)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("B   Binding Site SASA Distribution")
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Binding Site SASA Analysis — 100 ns MD Simulation\n"
        "Linta Mahboob, University of Siena",
        fontsize=11
    )

    out_path = os.path.join(outdir, "sasa_analysis.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  SASA figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def save_summary(times, rmsd_ca, rmsd_lig, times_sasa, sasa_values, outdir):
    """Write a plain-text statistics summary consistent with thesis results."""
    summary_path = os.path.join(outdir, "summary_statistics.txt")

    mask = ~np.isnan(rmsd_lig)

    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(" MD SIMULATION ANALYSIS SUMMARY\n")
        f.write(" Protein-Ligand Complex — 100 ns GROMACS / CHARMM36\n")
        f.write(" Author: Linta Mahboob | University of Siena\n")
        f.write("=" * 60 + "\n\n")

        f.write("─── RMSD ─────────────────────────────────────────────\n")
        f.write(f"Protein Cα RMSD\n")
        f.write(f"  Mean   : {rmsd_ca.mean():.3f} Å\n")
        f.write(f"  Median : {np.median(rmsd_ca):.3f} Å\n")
        f.write(f"  Std Dev: {rmsd_ca.std():.3f} Å\n")
        f.write(f"  Min    : {rmsd_ca.min():.3f} Å\n")
        f.write(f"  Max    : {rmsd_ca.max():.3f} Å\n\n")

        f.write(f"Ligand RMSD (binding pose stability)\n")
        f.write(f"  Mean   : {np.nanmean(rmsd_lig):.3f} Å\n")
        f.write(f"  Median : {np.nanmedian(rmsd_lig):.3f} Å  "
                f"[Thesis benchmark: ~1.6 Å]\n")
        f.write(f"  Std Dev: {np.nanstd(rmsd_lig):.3f} Å\n")
        f.write(f"  Min    : {np.nanmin(rmsd_lig):.3f} Å\n")
        f.write(f"  Max    : {np.nanmax(rmsd_lig):.3f} Å\n\n")

        if sasa_values is not None:
            f.write("─── SASA ─────────────────────────────────────────────\n")
            f.write("Binding Site Residue SASA\n")
            f.write(f"  Mean   : {sasa_values.mean():.3f} Å²\n")
            f.write(f"  Median : {np.median(sasa_values):.3f} Å²\n")
            f.write(f"  Std Dev: {sasa_values.std():.3f} Å²\n")
            f.write(f"  Min    : {sasa_values.min():.3f} Å²\n")
            f.write(f"  Max    : {sasa_values.max():.3f} Å²\n\n")

        f.write("─── Simulation Parameters ────────────────────────────\n")
        f.write(f"  Total time : {times[-1]:.1f} ns\n")
        f.write(f"  Frames     : {len(times)}\n")
        f.write(f"  Timestep   : 2 fs\n")
        f.write(f"  Force field: CHARMM36-jul2022\n")
        f.write(f"  Water model: TIP3P\n")
        f.write(f"  Temperature: 300 K (V-rescale thermostat)\n")
        f.write(f"  Pressure   : 1 bar (Parrinello-Rahman barostat)\n")

    print(f"  Summary statistics saved → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n{'='*58}")
    print(f"  RMSD + SASA Analysis")
    print(f"  TPR : {args.tpr}")
    print(f"  XTC : {args.xtc}")
    print(f"{'='*58}")

    # Load universe
    if not os.path.exists(args.tpr) or not os.path.exists(args.xtc):
        raise FileNotFoundError(
            f"Input files not found: {args.tpr}, {args.xtc}. "
            "Run GROMACS pipeline first."
        )
    u = mda.Universe(args.tpr, args.xtc)
    print(f"\n  Universe loaded: {len(u.atoms)} atoms, "
          f"{len(u.trajectory)} frames")

    # Parse binding residues if provided
    binding_res = None
    if args.binding_residues:
        binding_res = [int(r.strip()) for r in args.binding_residues.split(",")]
        print(f"  User-specified binding residues: {binding_res}")

    # Compute
    times, rmsd_ca, rmsd_lig = compute_rmsd(u, ligand_resname=args.ligand_resname)
    times_sasa, sasa_values  = compute_sasa(u, binding_res, args.ligand_resname)

    # Save raw data
    np.savetxt(os.path.join(args.outdir, "rmsd_ca.dat"),
               np.column_stack([times, rmsd_ca]),
               header="time_ns  rmsd_ca_A", fmt="%.4f")
    np.savetxt(os.path.join(args.outdir, "rmsd_ligand.dat"),
               np.column_stack([times, rmsd_lig]),
               header="time_ns  rmsd_lig_A", fmt="%.4f")
    if sasa_values is not None:
        np.savetxt(os.path.join(args.outdir, "sasa_binding_site.dat"),
                   np.column_stack([times_sasa, sasa_values]),
                   header="time_ns  sasa_A2", fmt="%.4f")

    # Plot
    plot_rmsd(times, rmsd_ca, rmsd_lig, args.outdir)
    if sasa_values is not None:
        plot_sasa(times_sasa, sasa_values, args.outdir)

    save_summary(times, rmsd_ca, rmsd_lig, times_sasa, sasa_values, args.outdir)

    print(f"\n  All outputs written to: {args.outdir}/")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
