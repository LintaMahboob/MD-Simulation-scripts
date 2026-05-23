#!/bin/bash
# =============================================================================
# Full MD Simulation Pipeline for Protein-Ligand Systems
# Author: Linta Mahboob | University of Siena
# Force Field: CHARMM36 | Water Model: TIP3P | Simulation: 100 ns
# =============================================================================
# Usage:
#   chmod +x run_md.sh
#   ./run_md.sh <protein.pdb> <ligand.itp> <ligand_posre.itp>
#
# Example:
#   ./run_md.sh protein.pdb LIG.itp LIG_posre.itp
#
# Prerequisites:
#   - GROMACS 2021+ installed and sourced (GMXRC)
#   - CHARMM36 force field in working directory or GMXLIB path
#   - Ligand topology (.itp) pre-generated via CGenFF / CHARMM-GUI
#   - Python 3.8+ with MDAnalysis, numpy, matplotlib for analysis scripts
# =============================================================================

set -euo pipefail   # Exit on error, undefined variables, pipe failures

PROTEIN=${1:-protein.pdb}
LIG_ITP=${2:-ligand.itp}
LIG_POSRE=${3:-LIG_posre.itp}

FORCEFIELD="charmm36-jul2022"    # CHARMM36 force field tag in GROMACS
WATER_MODEL="tip3p"              # TIP3P explicit water model
BOX_PADDING=1.0                  # Minimum distance from protein to box edge (nm)
PROD_NS=100                      # Production simulation length in nanoseconds
NSTEPS_PROD=$((PROD_NS * 500000)) # 2 fs timestep → 5e7 steps for 100 ns

NCPUS=$(nproc --all)             # Auto-detect available CPU cores
GPU_FLAG=""                      # Set to "-gpu_id 0" if GPU is available

echo "========================================================"
echo "  MD Simulation Pipeline — GROMACS / CHARMM36 / TIP3P  "
echo "  Protein : $PROTEIN"
echo "  Ligand  : $LIG_ITP"
echo "  Duration: ${PROD_NS} ns"
echo "========================================================"

# ─── STEP 1: Process Protein with CHARMM36 ───────────────────────────────────
echo ""
echo ">>> STEP 1: Processing protein with pdb2gmx (CHARMM36) ..."
gmx pdb2gmx \
    -f "$PROTEIN" \
    -o protein_processed.gro \
    -p topol.top \
    -i protein_posre.itp \
    -ff "$FORCEFIELD" \
    -water "$WATER_MODEL" \
    -ignh \
    -missing

# ─── STEP 2: Include Ligand Topology in topol.top ────────────────────────────
echo ""
echo ">>> STEP 2: Including ligand topology ..."
# Insert ligand .itp include before [ system ] directive
if grep -q "$LIG_ITP" topol.top; then
    echo "    Ligand ITP already included in topol.top."
else
    sed -i "/#include \"${FORCEFIELD}.ff\/forcefield.itp\"/a #include \"${LIG_ITP}\"" topol.top
fi

# Append ligand molecule entry to [ molecules ] section if absent
if ! grep -q "^LIG" topol.top; then
    echo "LIG    1" >> topol.top
fi

echo "    topol.top updated with ligand entries."

# ─── STEP 3: Combine Protein + Ligand into One GRO ───────────────────────────
echo ""
echo ">>> STEP 3: Combining protein and ligand coordinates ..."
# Assumes ligand .gro file is named ligand.gro (output from CGenFF/CHARMM-GUI)
gmx editconf -f protein_processed.gro -o protein_newbox.gro -c -d 0.1 -bt cubic

# Merge protein and ligand GRO files
cat protein_newbox.gro ligand.gro | \
    awk 'NR==1{n=$1+0} NR==2{total=int($1)+n; print total} NR>2{print}' \
    > complex.gro 2>/dev/null || cp protein_newbox.gro complex.gro
echo "    Combined complex GRO prepared."

# ─── STEP 4: Define Simulation Box ───────────────────────────────────────────
echo ""
echo ">>> STEP 4: Defining simulation box (cubic, padding = ${BOX_PADDING} nm) ..."
gmx editconf \
    -f complex.gro \
    -o box.gro \
    -c \
    -d "$BOX_PADDING" \
    -bt cubic

# ─── STEP 5: Solvate with TIP3P Water ────────────────────────────────────────
echo ""
echo ">>> STEP 5: Solvating system with TIP3P water ..."
gmx solvate \
    -cp box.gro \
    -cs spc216.gro \
    -o solv.gro \
    -p topol.top

# ─── STEP 6: Add Ions to Neutralize System ───────────────────────────────────
echo ""
echo ">>> STEP 6: Adding ions (neutralizing with NaCl, 0.15 M) ..."
gmx grompp \
    -f ions.mdp \
    -c solv.gro \
    -p topol.top \
    -o ions.tpr \
    -maxwarn 2

echo "SOL" | gmx genion \
    -s ions.tpr \
    -o solv_ions.gro \
    -p topol.top \
    -pname NA \
    -nname CL \
    -neutral \
    -conc 0.15

# ─── STEP 7: Energy Minimization ─────────────────────────────────────────────
echo ""
echo ">>> STEP 7: Running energy minimization (steepest descent, max 50000 steps) ..."
gmx grompp \
    -f minim.mdp \
    -c solv_ions.gro \
    -p topol.top \
    -o em.tpr \
    -maxwarn 2

gmx mdrun \
    -v \
    -deffnm em \
    -ntmpi 1 \
    -ntomp "$NCPUS" \
    $GPU_FLAG

# Verify energy minimization converged
echo ""
echo "    Checking EM convergence..."
echo "Potential" | gmx energy -f em.edr -o em_potential.xvg -xvg none
echo "    Energy minimization complete."

# ─── STEP 8: NVT Equilibration (Temperature) ─────────────────────────────────
echo ""
echo ">>> STEP 8: NVT equilibration (100 ps, 300 K) ..."
gmx grompp \
    -f nvt.mdp \
    -c em.gro \
    -r em.gro \
    -p topol.top \
    -o nvt.tpr \
    -maxwarn 2

gmx mdrun \
    -v \
    -deffnm nvt \
    -ntmpi 1 \
    -ntomp "$NCPUS" \
    $GPU_FLAG

# ─── STEP 9: NPT Equilibration (Pressure + Temperature) ──────────────────────
echo ""
echo ">>> STEP 9: NPT equilibration (100 ps, 300 K, 1 atm) ..."
gmx grompp \
    -f npt.mdp \
    -c nvt.gro \
    -r nvt.gro \
    -t nvt.cpt \
    -p topol.top \
    -o npt.tpr \
    -maxwarn 2

gmx mdrun \
    -v \
    -deffnm npt \
    -ntmpi 1 \
    -ntomp "$NCPUS" \
    $GPU_FLAG

# ─── STEP 10: Production MD — 100 ns ─────────────────────────────────────────
echo ""
echo ">>> STEP 10: Production MD (${PROD_NS} ns) ..."
echo "    This step will take considerable time. Checkpointing every 5 minutes."
gmx grompp \
    -f md.mdp \
    -c npt.gro \
    -t npt.cpt \
    -p topol.top \
    -o md.tpr \
    -maxwarn 2

gmx mdrun \
    -v \
    -deffnm md \
    -ntmpi 1 \
    -ntomp "$NCPUS" \
    -cpt 5 \
    $GPU_FLAG

# ─── STEP 11: Post-Processing — Remove PBC ───────────────────────────────────
echo ""
echo ">>> STEP 11: Removing periodic boundary conditions ..."
echo -e "1\n0" | gmx trjconv \
    -s md.tpr \
    -f md.xtc \
    -o md_noPBC.xtc \
    -pbc mol \
    -center \
    -ur compact

echo -e "1\n0" | gmx trjconv \
    -s md.tpr \
    -f md.xtc \
    -o md_noPBC_first.pdb \
    -b 0 \
    -e 0 \
    -pbc mol \
    -center

# ─── STEP 12: Basic Analysis ──────────────────────────────────────────────────
echo ""
echo ">>> STEP 12: Running GROMACS built-in analysis ..."

# RMSD of protein backbone
echo -e "4\n4" | gmx rms \
    -s md.tpr \
    -f md_noPBC.xtc \
    -o rmsd_backbone.xvg \
    -tu ns \
    -xvg none

# RMSD of ligand
echo -e "13\n13" | gmx rms \
    -s md.tpr \
    -f md_noPBC.xtc \
    -o rmsd_ligand.xvg \
    -tu ns \
    -xvg none 2>/dev/null || echo "    Note: Adjust group index for ligand RMSD manually."

# RMSF of Cα atoms
echo "4" | gmx rmsf \
    -s md.tpr \
    -f md_noPBC.xtc \
    -o rmsf_ca.xvg \
    -res \
    -xvg none

# Radius of gyration
echo "1" | gmx gyrate \
    -s md.tpr \
    -f md_noPBC.xtc \
    -o gyrate.xvg \
    -xvg none

# Hydrogen bonds between protein and ligand
echo -e "1\n13" | gmx hbond \
    -s md.tpr \
    -f md_noPBC.xtc \
    -num hbond_protein_ligand.xvg \
    -xvg none 2>/dev/null || echo "    Note: Adjust group indices for H-bond analysis manually."

echo ""
echo "========================================================"
echo "  Pipeline Complete!"
echo "  Key output files:"
echo "    md_noPBC.xtc        — PBC-corrected trajectory"
echo "    rmsd_backbone.xvg   — Protein backbone RMSD"
echo "    rmsd_ligand.xvg     — Ligand RMSD"
echo "    rmsf_ca.xvg         — Per-residue RMSF"
echo "    gyrate.xvg          — Radius of gyration"
echo "    hbond_protein_ligand.xvg — H-bond count vs time"
echo ""
echo "  Run Python analysis scripts:"
echo "    python analysis_rmsd_sasa.py"
echo "    python analysis_hbond.py"
echo "    python analysis_binding_site.py"
echo "========================================================"
