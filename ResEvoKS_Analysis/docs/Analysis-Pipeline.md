# Analysis Pipeline

End-to-end data flow, what every source file does, and how the scripts compose
them.

## Input contract

The analysis reads the artifacts the simulation writes, one file per evaluated
reservoir:

```
<results>/RUN<k>/matfiles/<gen>_<N>_<degree>_<radius>_<sigma>.{mat,jld2}
```

Each file holds `A`, `w_in`, `w_out`, `resparams` (`N`, `radius`, `degree`,
`sigma`, `beta`, `num_inputs`), `err` (`NRMSE` vector, `NMAE` scalar), and `J`.
`.mat` (MATLAB) and `.jld2` (native Julia) are read transparently; the dataset
files `data.{mat,jld2}` and any `history.*` are excluded. The generation is the
integer filename prefix.

## Data flow

```
                    results/RUN<k>/matfiles/*.{mat,jld2}
                                   │
                          ┌────────┴─────────┐
                          ▼                  ▼
                     DataAccess          (J, N) lists
                  load_record / A             │
                          │                   ▼
                      Sampling ───────────►  Pareto         (size–efficiency front)
              select_generations +              fit  f(x)=a e^{-bx}+c
              stratified_sample                 │
                          │                     ▼
                          ▼                  ErrorStats     (per-gen log10 J)
                 sampled reservoirs (A)
                          │
        ┌─────────────────┼──────────────────────────┐
        ▼                 ▼                           ▼
     Spectral          Modularity                MultiObjective
  rw_laplacian      detect_communities         normalize_metrics
  eigenvalues       directed_modularity        composite_objectives
  smoothed_density  connection_cost            run_nsga2 (NSGA-II)
        │                 │                     closest_observed
        ▼                 └──────────► (modularity, cost, J, gen) ──┘
    Embedding
  interpolate + PCA + k-means
        │
        ▼
   SpectralEMD
  spectral_emd / pairwise / to-reference

   ReferenceGraphs  (independent: ER/BA/WS/SBM spectra for the envelope)
```

## Source files

| File | Role |
|------|------|
| `src/ResEvoKS_Analysis.jl` | top module; `include`s the submodules and re-exports the public API. |
| `src/DataAccess.jl` | read run directories; list/filter artifact files; parse generations; `load_record`, `load_adjacency`, `read_J_N`, `collect_J_N`. Format-agnostic (`.mat`/`.jld2`). |
| `src/Sampling.jl` | `find_suitable_last_gen`, `select_generations` (evenly spaced across evolution), `stratified_sample` / `stratified_sample_run` (quartile sampling, seeded). |
| `src/Pareto.jl` | `pareto_frontier` (2-D non-domination), `fit_exponential` (`f(x)=a e^{-bx}+c`, with `R²`). |
| `src/Spectral.jl` | `rw_laplacian` (`I−D⁻¹A`), `laplacian_eigenvalues` (ARPACK + dense fallback), `smoothed_density`, `spectral_density_grid`, `spectral_centroid`. |
| `src/Embedding.jl` | `interpolate_spectrum`, `spectra_matrix`, `spectra_pca` (PCA + explained variance), `cluster_spectra` (k-means, silhouette-selected `k`). |
| `src/SpectralEMD.jl` | `spectral_emd` (optimal transport, squared-Euclidean cost), `pairwise_emd`, `emd_to_reference`. |
| `src/Modularity.jl` | `detect_communities` (label propagation), `directed_modularity` (Newman `Q`), `connection_density`, `average_path_length` (Floyd–Warshall), `connection_cost`, `network_metrics`. |
| `src/MultiObjective.jl` | `normalize_metric(s)`, `composite_objectives` (`O₁…O₄`), `run_nsga2` (NSGA-II), `closest_observed`. |
| `src/ReferenceGraphs.jl` | `er_spectrum`, `ba_spectrum`, `ws_spectrum`, `sbm_spectrum`, `reference_spectra` — canonical-ensemble spectra under the same RW Laplacian. |
| `src/ErrorStats.jl` | `collect_generation_errors`, `log_error_distribution`, `generation_error_stats`. |

## Scripts (figures)

Each script is a thin orchestrator: it reads, calls the pure functions, and
renders with `Plots.jl`. All are parameterized by `RESEVO_*` environment
variables (see [Running & Reproducibility](Running-and-Reproducibility.md)).

| Script | Produces | Modules used |
|--------|----------|--------------|
| `scripts/run_pareto.jl` | size–efficiency scatter + frontier + exponential fit per run/gen | `DataAccess`, `Pareto` |
| `scripts/run_spectral.jl` | PCA by generation, k-means clusters, spectral-density averages + heatmap, EMD drift | `Sampling`, `Spectral`, `Embedding`, `SpectralEMD` |
| `scripts/run_modularity.jl` | modularity-vs-cost scatter, per-gen means, cost decay fit, NSGA-II trade-off | `Sampling`, `Modularity`, `Pareto`, `MultiObjective` |
| `scripts/run_error_distributions.jl` | per-generation `log10 J` histograms + spread | `ErrorStats` |
| `scripts/run_reference_spectra.jl` | reference-ensemble spectral densities (single + averaged) | `ReferenceGraphs`, `Spectral` |

## What was deliberately not ported

The original scripts contained large free-text "comprehensive report" functions
that printed paragraphs of interpretation to disk. These were console narration,
not method, and are omitted. Where a saved summary is useful, the port returns a
tidy struct (e.g. `GenerationErrorStats`, `NetworkMetrics`, `ParetoFit`) that a
caller can serialize compactly.
