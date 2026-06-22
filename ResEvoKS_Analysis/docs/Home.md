# ResEvoKS_Analysis Wiki — Home

Documentation for **ResEvoKS_Analysis**, the Phase-2 analysis suite for the
evolutionary reservoir-computing project. It consumes the reservoirs evolved by
[`ResEvoKS_Simulation`](../../ResEvoKS_Simulation) and reproduces the structural analyses of the paper

> *Evolutionary Optimization Reveals Structural Constraints on Reservoir
> Architecture for Spatiotemporal Chaos*, N. Dehghani.

This wiki accompanies the [README](../README.md) and the
[original-code report](../REPORT_analysis_code.md).

## What this code does, in one paragraph

The simulation evolves echo-state reservoirs and saves *every* evaluated
reservoir — its recurrent matrix `A`, its composite forecast error `J`, and its
hyperparameters. This package reads those saved reservoirs and characterizes the
**structure** that evolution selects for. It measures how predictive efficiency
trades off against reservoir size (a Pareto front), summarizes each reservoir's
connectivity by the spectrum of its random-walk Laplacian (then compares spectra
across generations by PCA, smoothed density, and optimal-transport distance),
quantifies community modularity and wiring cost, and runs a post-hoc NSGA-II to
map the performance–structure trade-off. Every measurement is a pure function of
the saved `A` / `J`; the scripts turn them into the paper's figures.

## Wiki pages

| Page | Contents |
|------|----------|
| **[Mathematics](Mathematics.md)** | The size–efficiency Pareto front and exponential fit; the random-walk Laplacian `L=I−D⁻¹A` and its spectrum; the smoothed density `Γ`; fixed-length interpolation + PCA; the optimal-transport spectral distance (EMD); directed Newman modularity; density / path length / connection cost; the four NSGA-II objectives. Each block links to the function that implements it. |
| **[Analysis Pipeline](Analysis-Pipeline.md)** | End-to-end data flow; what every source file and function does; how the scripts compose them; the on-disk input contract. |
| **[API Reference](API-Reference.md)** | Every exported type and function, with signatures and arguments. |
| **[Running & Reproducibility](Running-and-Reproducibility.md)** | Install; run each analysis script; seeds and determinism; ARPACK vs KrylovKit; scaling and runtime notes. |

## Map: paper ↔ code

| Paper Methods subsection                            | Equation(s) / idea                          | Implemented in |
|-----------------------------------------------------|---------------------------------------------|----------------|
| Reservoir size–efficiency trade-off                 | Pareto front, `f(x)=a e^{-bx}+c`            | `Pareto` |
| Network selection for spectral analysis             | evenly spaced gens, quartile sampling       | `Sampling` |
| Normalized random-walk Laplacian spectrum           | `L=I−D⁻¹A`, ARPACK eigenvalues              | `Spectral` |
| Smoothed eigenvalue distributions                   | `Γ(x)=1/m Σ N(x;λ_i,s²)`                     | `Spectral` |
| PCA of Laplacian spectra                            | sort+interpolate to `q`, PCA on `S`         | `Embedding` |
| Spectral distance using optimal transport           | EMD, squared-Euclidean cost                 | `SpectralEMD` |
| Community detection and modularity                  | label propagation, directed `Q`            | `Modularity` |
| Connection density, path length, connection cost    | `density`, `ℓ`, `C=α Σ|A|+β·density+γ·ℓ`    | `Modularity` |
| Multi-objective analysis                            | `O₁…O₄`, NSGA-II                            | `MultiObjective` |
| SBM-like spectral envelope                          | ER / BA / WS / SBM reference spectra        | `ReferenceGraphs` |
| Population-level error reduction                    | per-generation `log₁₀ J` distributions      | `ErrorStats` |

> The simulation half is documented in the companion
> [`ResEvoKS_Simulation` wiki](../../ResEvoKS_Simulation/docs/Home.md). This wiki documents the
> **analysis** half.
