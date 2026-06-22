# Running & Reproducibility

## Install

Requires **Julia ≥ 1.9**. From `julia/ResEvoKS_Analysis`:

```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

If you edit `Project.toml`, run `Pkg.resolve()` then `Pkg.instantiate()` again.
The first load precompiles a fairly heavy stack (Graphs, OptimalTransport,
Tulip, Metaheuristics, MultivariateStats, ARPACK, Plots) — expect a few minutes
once.

Run the tests (synthetic fixtures; no simulation output needed):

```bash
julia --project=. test/runtests.jl
```

The suite fabricates a small `results/RUN1/matfiles` tree matching the
simulation's on-disk contract and exercises every analysis module through it.

---

## Run the analyses

All scripts share the `RESEVO_*` environment-variable convention and write
figures under `RESEVO_OUTDIR` (default `analysis_results/<analysis>/RUN<k>/`).
Point `RESEVO_RESULTS` at the directory that contains the simulation's `RUN<k>`
folders.

```bash
# Size–efficiency Pareto fronts
RESEVO_RESULTS=../ResEvoKS_Simulation/results RESEVO_RUNS=1:4 RESEVO_GENS=0:10:50 \
  julia --project=. scripts/run_pareto.jl

# Laplacian spectra: PCA, clustering, density evolution, EMD drift
RESEVO_RESULTS=../ResEvoKS_Simulation/results RESEVO_RUN=1 \
  julia --project=. scripts/run_spectral.jl

# Modularity, connection cost, NSGA-II
RESEVO_RESULTS=../ResEvoKS_Simulation/results RESEVO_RUN=1 RESEVO_NSGA=1 \
  julia --project=. scripts/run_modularity.jl

# Population-level error reduction
RESEVO_RESULTS=../ResEvoKS_Simulation/results RESEVO_RUNS=1:4 \
  julia --project=. scripts/run_error_distributions.jl

# SBM-like reference envelope (no simulation data required)
julia --project=. scripts/run_reference_spectra.jl
```

| Variable           | Used by                       | Meaning                                   | Default |
|--------------------|-------------------------------|-------------------------------------------|---------|
| `RESEVO_RESULTS`   | all (except reference)        | root holding `RUN<k>/matfiles`            | `results` |
| `RESEVO_RUN`       | spectral, modularity          | single run index                          | `1` |
| `RESEVO_RUNS`      | pareto, errors                | run index range                           | `1:10` |
| `RESEVO_GENS`      | pareto, errors                | generations to analyze                    | `0:10:50` |
| `RESEVO_NSELECT`   | spectral, modularity          | generations across evolution              | `10` / `20` |
| `RESEVO_NQUARTILE` | spectral, modularity          | reservoirs per quartile                   | `20` |
| `RESEVO_TARGETQ`   | spectral                      | interpolation length `q` for PCA          | `200` |
| `RESEVO_NSGA`      | modularity                    | run NSGA-II (`1`) or skip (`0`)           | `1` |
| `RESEVO_N`         | reference                     | reference-graph size `n`                  | `500` |
| `RESEVO_NINST`     | reference                     | instantiations to average                 | `100` |
| `RESEVO_OUTDIR`    | all                           | figure output root                        | `analysis_results` |

---

## Reproducibility

- **Sampling.** Generation selection is deterministic; the quartile draw inside
  `stratified_sample` uses an explicit `StableRNG` (default seed `1234`, as in
  the original `Random.seed!(1234)`). A given seed reproduces the same sampled
  reservoirs across Julia versions.
- **Clustering.** `cluster_spectra` accepts an `rng`, so k-means labels are
  reproducible when seeded.
- **Reference graphs.** Each generator takes a `StableRNG`; the scripts vary the
  seed per instantiation so the averaged spectra are reproducible.
- **NSGA-II.** `run_nsga2` is stochastic; the Pareto set varies run-to-run, but
  the matched closest-observed reservoirs are stable in aggregate. Seed
  Metaheuristics' global RNG before calling if exact reproduction is required.

Two caveats inherited from the data: reservoirs with `J = Inf` (no channel beat
the NRMSE threshold) are non-finite in error analyses and are dropped from
`log10 J` distributions; and a reservoir with ill-conditioned spectrum falls
back to a dense eigensolve (a warning is emitted).

---

## ARPACK vs. KrylovKit

The Laplacian spectrum uses **ARPACK** (`Arpack.eigs`), as mentioned in the paper. The
*simulation* deliberately avoids ARPACK for its spectral-radius step because
ARPACK is not thread-safe and the GA evaluates reservoirs across threads. The
analysis runs **single-threaded**, so ARPACK is safe here and is used for
fidelity to the published method. For tiny matrices (`n ≤ 20`) or non-convergent
cases the code falls back to a dense `eigvals`.

---

## Runtime & memory

Per analysis the cost is dominated by:

- **Eigenvalues:** one sparse `eigs` per sampled reservoir
  (`O(nev · nnz(A))` per Arnoldi step). With ~80 reservoirs/generation × ~10
  generations this is the bulk of `run_spectral.jl`.
- **Pairwise EMD:** `pairwise_emd` is `O(M²)` exact-OT solves — use
  `emd_to_reference` (linear in `M`) for large samples, as the original did. Heavy computation!
- **Path length:** `average_path_length` runs Floyd–Warshall, `O(n_r³)` per
  reservoir — the dominant cost in `run_modularity.jl` for large `n_r`.
- **NSGA-II:** `N=1000` over a few hundred generations; seconds to minutes. Rakes up compute minutes for large population.

Reduce `RESEVO_NQUARTILE` / `RESEVO_NSELECT` to shrink the sample, or restrict
`RESEVO_RUNS` / `RESEVO_GENS`, when iterating.
