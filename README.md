# ResEvoKS — Evolutionary Reservoir Computing for Spatiotemporal Chaos
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20807393.svg)](https://doi.org/10.5281/zenodo.20807393)
[![Paper Card](https://img.shields.io/badge/Paper%20Card-Neurovium-6f42c1)](https://neurovium.science/posts/Adaptive-reservoir)
[![Blog Post](https://img.shields.io/badge/Blog%20Post-Neurovium-0a7cff)](https://neurovium.science/posts/pblog-Adaptive-reservoir)

Companion code for the paper

> **Evolutionary Optimization Reveals Structural Constraints on Reservoir
> Architecture for Spatiotemporal Chaos**, Nima Dehghani. 2026.
[![arXiv](https://img.shields.io/badge/arXiv-2606.22765-b31b1b.svg)](https://arxiv.org/abs/2606.22765)
[![PDF](https://img.shields.io/badge/PDF-2606.22765-blue.svg)](https://arxiv.org/pdf/2606.22765)
[![HTML](https://img.shields.io/badge/HTML-2606.22765-green.svg)](https://arxiv.org/html/2606.22765v1)

This investigation asks a simple yet foundational question with a large search: *when you evolve echo-state **reservoir computers** to forecast the
chaotic **Kuramoto–Sivashinsky (KS)** equation, what kind of recurrent network
structure does prediction actually select for?* The work has two halves —
**evolving** the reservoirs, and **dissecting** the ones that survive — and this
repository ships them as two self-contained Julia packages.

---

## A note on the port: now fully Julia

The project grew up polyglot:

- the **simulation** (KS solver, reservoir construction, training, and the
  parallel genetic algorithm) was originally written in **MATLAB**;
- a few **light demos** — the schematic "good vs. bad reservoir" picture and a
  reference-graph spectral sketch — were in **Python**;
- the **analysis** (Pareto front, Laplacian spectra, modularity, multi-objective
  trade-offs) was already in **Julia**.

This repository consolidates all of that into **one fully-Julia code suite**: a
faithful, modular, well-documented port of the simulation, plus a cleaned-up and
modularized rewrite of the analysis. The MATLAB and Python originals are no
longer needed to run anything here. The on-disk data contract is preserved, so
the two packages interoperate exactly as the original MATLAB→Julia workflow did.

---

## The two packages

| Package | Role | Start here |
|---------|------|-----------|
|**[`ResEvoKS_Simulation/`](ResEvoKS_Simulation)** | Evolves reservoirs to predict KS chaos and writes every evaluated reservoir to disk. (Port of the MATLAB `64D-KS-Sim` suite + the Python demo.) | [README](ResEvoKS_Simulation/README.md) |
|**[`ResEvoKS_Analysis/`](ResEvoKS_Analysis)** | Reads those saved reservoirs and produces the paper's structural figures and statistics. (Cleaned-up rewrite of the Julia analysis scripts + the Python reference-graph sketch.) | [README](ResEvoKS_Analysis/README.md) |

The packages are **deliberately independent** Julia environments. The analysis
does not depend on the simulation as a library — it simply reads the artifacts
the simulation leaves on disk:

```
ResEvoKS_Simulation  ──writes──▶  <results>/RUN<k>/matfiles/<gen>_<N>_<deg>_<rad>_<sig>.{mat,jld2}  ──reads──▶  ResEvoKS_Analysis
```

Each artifact holds the recurrent matrix `A`, input/readout weights
`w_in`/`w_out`, the construction parameters `resparams`, the prediction error
`err`, and the composite fitness `J`. Both `.mat` (MATLAB-compatible) and `.jld2`
(native Julia) formats are supported, and the analysis reads either.

---

## What each half does

### Simulation — `ResEvoKS_Simulation`

A genetic algorithm searches reservoir *construction* hyperparameters (size,
connectivity degree, spectral radius, input scaling, ridge regularization). Each
candidate reservoir is built, trained with a ridge readout, and scored by how
long and how accurately it forecasts the KS field in closed-loop (autonomous)
mode. Highlights of the port:

- a **stable ETDRK4 KS solver**;
- a **self-contained generational GA** with thread-level parallelism;
- a **bit-compatible output contract** so artifacts round-trip with the analysis.

→ Details: [Home](ResEvoKS_Simulation/docs/Home.md) ·
[Mathematics](ResEvoKS_Simulation/docs/Mathematics.md) ·
[Pipeline](ResEvoKS_Simulation/docs/Simulation-Pipeline.md) ·
[API](ResEvoKS_Simulation/docs/API-Reference.md) ·
[Running & Reproducibility](ResEvoKS_Simulation/docs/Running-and-Reproducibility.md) ·

### Analysis — `ResEvoKS_Analysis`

Given a population of evolved reservoirs, the analysis characterizes the
structure prediction selects for:

- the **size–efficiency Pareto front** (how small a reservoir can be and still
  predict well);
- the **normalized random-walk Laplacian spectrum**, its smoothed densities,
  PCA embedding, and optimal-transport (EMD) distances;
- **community modularity, connection density, path length, and connection cost**;
- an **NSGA-II multi-objective** trade-off over performance, modularity, and cost;
- **reference-graph envelopes** (ER / BA / WS / SBM) for comparison.

Every `src/` module is pure computation (unit-tested); the `scripts/` render the
figures with `Plots.jl`.

→ Details: [Home](ResEvoKS_Analysis/docs/Home.md) ·
[Mathematics](ResEvoKS_Analysis/docs/Mathematics.md) ·
[Pipeline](ResEvoKS_Analysis/docs/Analysis-Pipeline.md) ·
[API](ResEvoKS_Analysis/docs/API-Reference.md) ·
[Running & Reproducibility](ResEvoKS_Analysis/docs/Running-and-Reproducibility.md) ·

---

## Quick start

Both packages require **Julia ≥ 1.9** and manage their own dependencies. Run the
simulation first to generate reservoirs, then point the analysis at the results.

```bash
# 1. Evolve reservoirs (small, fast configuration)
cd ResEvoKS_Simulation
julia --project=. -e 'using Pkg; Pkg.instantiate()'
RESEVO_NRUNS=1 RESEVO_NSTEP=20000 RESEVO_POP=40 RESEVO_GENS=15 \
  julia --project=. -t auto scripts/run_optimization.jl
# → writes results/RUN1/matfiles/*.mat

# 2. Analyze the evolved reservoirs
cd ../ResEvoKS_Analysis
julia --project=. -e 'using Pkg; Pkg.instantiate()'
RESEVO_RESULTS=../ResEvoKS_Simulation/results \
  julia --project=. scripts/run_pareto.jl
```

See each package's README and its *Running & Reproducibility* page for the
paper-scale configuration, thread/BLAS guidance, and the full list of
`RESEVO_*` environment variables.

---

## Repository layout

```
julia/
├── README.md                  ← you are here (repo overview)
├── ResEvoKS_Simulation/       evolutionary KS reservoir simulation (Julia)
│   ├── src/  scripts/  test/  docs/
│   ├── README.md   
│   └── Project.toml
└── ResEvoKS_Analysis/         structural analysis of evolved reservoirs (Julia)
    ├── src/  scripts/  test/  docs/
    ├── README.md   
    └── Project.toml
```

---

## Citation

If you use this code, please cite **both the software and the paper**. The
repository ships a [`CITATION.cff`](CITATION.cff), so GitHub's *"Cite this
repository"* button and citation managers (Zenodo, Papers, etc.) can export the
reference automatically.

**The software:**

> Dehghani, Nima. *ResEvoKS: Evolutionary Reservoir Computing for Spatiotemporal
> Chaos (Julia)*, 2026. https://doi.org/10.5281/zenodo.20807393 

**The paper**:

```bibtex
@article{Dehghani2026resevomech,
      title={Evolutionary Optimization Reveals Structural Constraints on Reservoir Architecture for Spatiotemporal Chaos}, 
      author={Nima Dehghani},
      year={2026},
      eprint={2606.22765},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2606.22765}, 
}
```

[![Paper Card](https://img.shields.io/badge/Paper%20Card-Neurovium-6f42c1)](https://neurovium.science/posts/Adaptive-reservoir)
[![Blog Post](https://img.shields.io/badge/Blog%20Post-Neurovium-0a7cff)](https://neurovium.science/posts/pblog-Adaptive-reservoir)

```
**Nima Dehghani**.
