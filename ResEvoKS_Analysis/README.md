# ResEvoKS_Analysis

**Phase-2 analysis suite** for the evolutionary reservoir-computing project —
the Julia companion that turns the reservoirs evolved by
[`ResEvoKS_Simulation`](../ResEvoKS_Simulation) into the structural figures and statistics of the
paper

> *Evolutionary Optimization Reveals Structural Constraints on Reservoir
> Architecture for Spatiotemporal Chaos*, N. Dehghani.

`ResEvoKS_Simulation` (the simulation) evolves echo-state reservoirs to predict
Kuramoto–Sivashinsky chaos and saves every evaluated reservoir to disk.
`ResEvoKS_Analysis` (this package) reads those saved reservoirs and asks **what
kind of recurrent structure prediction selects for** — through the
size–efficiency Pareto front, the random-walk Laplacian spectrum, community
modularity and connection cost, and an NSGA-II multi-objective trade-off.

The port and its mapping to the original analysis code are documented in
[`REPORT_analysis_code.md`](REPORT_analysis_code.md); the mathematics and a full
file/function reference live in the [`docs/` wiki](docs/Home.md).

---

## What it computes (paper Methods → module)

| Analysis (paper Methods subsection)                 | Module           | Key functions |
|-----------------------------------------------------|------------------|---------------|
| Reservoir size–efficiency trade-off                 | `Pareto`         | `pareto_frontier`, `fit_exponential` |
| Network selection for spectral analysis             | `Sampling`       | `select_generations`, `stratified_sample` |
| Normalized random-walk Laplacian spectrum           | `Spectral`       | `rw_laplacian`, `laplacian_eigenvalues` |
| Smoothed eigenvalue distributions                   | `Spectral`       | `smoothed_density`, `spectral_density_grid` |
| PCA of Laplacian spectra                            | `Embedding`      | `interpolate_spectrum`, `spectra_pca`, `cluster_spectra` |
| Spectral distance using optimal transport (EMD)     | `SpectralEMD`    | `spectral_emd`, `pairwise_emd`, `emd_to_reference` |
| Community detection and modularity                  | `Modularity`     | `detect_communities`, `directed_modularity` |
| Connection density, path length, connection cost    | `Modularity`     | `connection_density`, `average_path_length`, `connection_cost` |
| Multi-objective analysis (NSGA-II)                  | `MultiObjective` | `normalize_metrics`, `composite_objectives`, `run_nsga2` |
| SBM-like reference envelope                          | `ReferenceGraphs`| `er_spectrum`, `ba_spectrum`, `ws_spectrum`, `sbm_spectrum` |
| Population-level error reduction                    | `ErrorStats`     | `collect_generation_errors`, `generation_error_stats` |
| Reading the simulation's saved artifacts            | `DataAccess`     | `run_matdir`, `load_record`, `load_adjacency`, `read_J_N` |

**Design:** every `src/` module is **pure computation** — it takes matrices and
returns numbers / arrays / small result structs, and never plots. The figures
are produced by the `scripts/`, which call these functions and render with
`Plots.jl`. This keeps the science unit-testable and the plotting stack out of
the core. See [the report](REPORT_analysis_code.md#3-design-of-the-julia-port)
for the rationale.

---

## Install

Requires **Julia ≥ 1.9**. From `julia/ResEvoKS_Analysis`:

```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

Run the tests (synthetic fixtures, no simulation data needed):

```bash
julia --project=. test/runtests.jl
```

---

## Input: the simulation's output

The analysis reads the per-individual artifacts written by `ResEvoKS_Simulation` under

```
<results>/RUN<k>/matfiles/<gen>_<N>_<degree>_<radius>_<sigma>.{mat,jld2}
```

each holding `A` (recurrent matrix after spectral scaling), `w_in`, `w_out`,
`resparams`, `err` (`NRMSE`, `NMAE`), and `J`. Both `.mat` and `.jld2` are read
transparently. Point the scripts at the `<results>` root (the directory that
contains the `RUN<k>` folders).

---

## Reproducing the paper analyses

Each script is parameterized by environment variables and writes figures under
`RESEVO_OUTDIR` (default `analysis_results`).

```bash
# Size–efficiency Pareto fronts (per run/generation)
RESEVO_RESULTS=results RESEVO_RUNS=1:4 RESEVO_GENS=0:10:50 \
  julia --project=. scripts/run_pareto.jl

# Random-walk Laplacian spectra: PCA, clustering, density evolution, EMD drift
RESEVO_RESULTS=results RESEVO_RUN=1 RESEVO_NSELECT=10 \
  julia --project=. scripts/run_spectral.jl

# Modularity, connection cost, and NSGA-II multi-objective trade-off
RESEVO_RESULTS=results RESEVO_RUN=1 RESEVO_NSELECT=20 RESEVO_NSGA=1 \
  julia --project=. scripts/run_modularity.jl

# Population-level error reduction across generations
RESEVO_RESULTS=results RESEVO_RUNS=1:4 RESEVO_GENS=0:10:50 \
  julia --project=. scripts/run_error_distributions.jl

# SBM-like reference envelope (canonical graph ensembles)
RESEVO_N=500 RESEVO_NINST=100 \
  julia --project=. scripts/run_reference_spectra.jl
```

| Variable           | Meaning                                   | Default            |
|--------------------|-------------------------------------------|--------------------|
| `RESEVO_RESULTS`   | root holding `RUN<k>/matfiles`            | `results`          |
| `RESEVO_RUN(S)`    | run index / range to analyze              | `1` / `1:4`        |
| `RESEVO_GENS`      | generations (Pareto / error scripts)      | `0:10:50`          |
| `RESEVO_NSELECT`   | generations selected across evolution     | `10` (spec) / `20` (mod) |
| `RESEVO_NQUARTILE` | reservoirs sampled per performance quartile | `20`             |
| `RESEVO_NSGA`      | run NSGA-II (`1`) or skip (`0`)           | `1`                |
| `RESEVO_OUTDIR`    | output directory for figures              | `analysis_results` |

---

## Programmatic use

```julia
using ResEvoKS_Analysis

matdir = run_matdir("results", 1)                       # results/RUN1/matfiles
gens   = available_generations(matdir)
sel    = select_generations(gens, matdir; n_select=10)  # ~10 gens across evolution
samp   = stratified_sample_run(matdir, sel)             # ≤80 reservoirs / gen

# random-walk Laplacian spectra of the sampled reservoirs
evs = [laplacian_eigenvalues(load_adjacency(matdir, f)) for f in samp]

# PCA + clustering of the spectra
S    = spectra_matrix(evs, 200)
pca  = spectra_pca(S; maxoutdim=3)
clus = cluster_spectra(S)

# structural metrics of one reservoir
m = network_metrics(load_adjacency(matdir, samp[1]))     # modularity, cost, ...
```

See the [API reference](docs/API-Reference.md) for every exported function.

---

## Notes

- **ARPACK** is used for the Laplacian spectrum (as in the paper). It is
  single-threaded here, so the thread-safety issue that pushed the *simulation*
  to KrylovKit does not apply. A dense fallback covers tiny / non-convergent
  cases.
- **Reproducibility:** stratified sampling and clustering take an explicit
  `StableRNG`, so a given seed reproduces the same sample and labels across
  Julia versions (the original used `Random.seed!(1234)`).
- This package is **standalone** — it does not depend on the simulation package;
  it only reads the on-disk artifacts.
