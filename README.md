# MD Simulation Pipeline for Protein-Ligand Interactions

Molecular dynamics workflow and analysis scripts used to study binding-site
dynamics across a set of 100 protein-ligand co-crystal structures.

**Author:** Linta Mahboob
**Force field:** CHARMM36-mar2019
**Water model:** TIP3P (explicit solvent)
**Simulation length:** 100 ns production MD per complex
**Software:** GROMACS 2019.3

---

## Dataset

100 protein-ligand complexes selected from the RCSB PDB, restricted to:
- unmutated, atomic-resolution X-ray structures
- soluble targets
- complexes with experimentally confirmed inhibitory activity
- ligands without ions or unusual functional groups requiring nonstandard parameterization

Structure and sequence metadata (UniProtKB accession, PDB code, method, resolution)
are listed in `structures/dataset_100_complexes.csv`.

## Repository structure

```
md_simulation_pipeline/
├── README.md
├── structures/
│   └── dataset_100_complexes.csv     # PDB/UniProt metadata for the 100-complex dataset
├── scripts/
│   ├── analysis_rmsd_sasa.py         # RMSD (protein, ligand) and binding-site SASA
│   ├── analysis_hbond.py             # H-bond occupancy analysis
│   └── analysis_binding_site.py      # Binding-site residue composition
└── data/
    └── RMSD.xlsx                     # Aggregated per-complex RMSD time series (ps, nm)
```

## Structure preparation

- Missing side chains and steric clashes were corrected by molecular modeling in PyMOD 3.0
- Sequence alignments: CLUSTALO 1.2.0
- Homology/modeling tool: Modeller v9.3
- No docking was performed; co-crystal poses were retained and only structurally optimized

## Simulation protocol

| Step | Tool | Details |
|---|---|---|
| Energy minimization | GROMACS 2019.3 | steepest descent, 5000 steps, convergence at F < 100 kJ/mol/nm |
| Solvation | `gmx solvate` | cubic box, TIP3P water |
| Neutralization | `gmx genion` | counterions added to neutralize net charge |
| Short equilibration | cMD | 10 ns, prior to PLIP-based binding-residue identification |
| Production MD | cMD | 100 ns, NPT |

- Thermostat: V-rescale, 300 K
- Barostat: Nosé-Hoover, 1 atm, damping 1 ps⁻¹
- Bond constraints: LINCS (H-containing bonds)
- Ligand topologies: CHARMM-GUI 3.8
- Binding-site residues: identified per complex with PLIP 2.3.0 following the short equilibration run

## Analyses

RMSD (protein backbone, binding-residue backbone, ligand) and SASA (binding
residues) were computed for each complex. Hydrogen bond occupancy between
binding residues and ligand was evaluated via the existence matrix (`gmx bond`,
`-hbn -hbm`), binned into three time windows over the 100 ns run: 0-30 ns
(low), 31-70 ns (moderate), 71-100 ns (high occupancy).

The scripts in `scripts/` reproduce this analysis in Python/MDAnalysis, for
use on individual complexes or as a starting point for batch processing:

```bash
python scripts/analysis_rmsd_sasa.py --tpr md.tpr --xtc md_noPBC.xtc --ligand-resname LIG
python scripts/analysis_hbond.py --tpr md.tpr --xtc md_noPBC.xtc --ligand-resname LIG
python scripts/analysis_binding_site.py --tpr md.tpr --xtc md_noPBC.xtc --ligand-resname LIG
```

All outputs (plots, `.csv`/`.dat` data, summary statistics) are written to `analysis_output/`.

### Requirements

```bash
pip install MDAnalysis numpy matplotlib pandas
pip install freesasa   # for accurate SASA calculations
```

## Key results

- Binding-residue backbone RMSD: median 1.2 Å (IQR 0.7-1.5 Å)
- Ligand RMSD: median 1.6 Å (IQR 1.0-2.0 Å)
- Binding-site SASA (minimum): median 2.68 Å² (IQR 2.29-2.72 Å²)
- Binding-site SASA (maximum): median 3.2 Å² (IQR 3.03-3.62 Å²)
- H-bond occupancy across all binding residues: 7.2% low, 6.3% moderate, 86.5% high
- Binding-site residue composition: charged residues 56% (Asp 28.1%, His 11.7%, Arg 9.2%
  individually most frequent), hydrophobic 25%, polar 19%
