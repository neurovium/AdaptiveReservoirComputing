# ResEvoKS_Simulation Wiki — Home

Welcome to the documentation for **ResEvoKS_Simulation**, the Julia simulation suite for
evolutionary reservoir computing on Kuramoto–Sivashinsky (KS) spatiotemporal
chaos. This wiki accompanies the [README](../README.md) and the
[original-code report](../REPORT_original_code.md).

## What this code does, in one paragraph

A reservoir computer is a fixed recurrent network (the *reservoir*) whose
high-dimensional transient response to an input signal is read out by a single
trained linear layer. Here the input is the chaotic KS field, the readout is
trained by ridge regression, and prediction is done in **closed loop** — the
network's own output is fed back as its next input. A **genetic algorithm**
searches the five hyperparameters that *construct* the reservoir (size,
connectivity degree, spectral radius, input scaling, ridge regularization),
scoring each candidate by a composite forecast-error metric `J`. The simulation
records every evaluated reservoir so the downstream analysis can ask *what kind
of recurrent substrate prediction selects for*.

## Wiki pages

| Page | Contents |
|------|----------|
| **[Mathematics](Mathematics.md)** | The KS equation and ETDRK4 integrator; reservoir state update; ridge readout and the bilinear feature map; autonomous prediction; NRMSE/NMAE and the composite fitness `J`; spectral-radius scaling; size–efficiency Pareto fit. Each block links the equation to the function that implements it. |
| **[Output & logging](Simulation-Pipeline.md#output-contract)** | The `.mat`/`.jld2` output formats and the per-run timing/failure log. |
| **[Simulation Pipeline](Simulation-Pipeline.md)** | End-to-end data flow; what every source file and function does; the GA driver design; the on-disk output contract. |
| **[API Reference](API-Reference.md)** | Every exported type and function, with signatures and arguments. |
| **[Running & Reproducibility](Running-and-Reproducibility.md)** | How to install and run; threads vs. BLAS; seeds and determinism; scaling the experiment up or down; runtime/memory expectations. |

## Map: paper ↔ code

| Paper Methods subsection | Equation(s) | Implemented in |
|--------------------------|-------------|----------------|
| KS time-series generation | KS PDE, `L_k = k²−k⁴`, ETDRK4 | `KuramotoSivashinsky.solve_ks` |
| Reservoir computing model | sparsity `p=d/n`, spectral scaling, `Win` blocks, `x(t)=tanh(Ax+Winu)` | `Reservoir.*`, `Readout.reservoir_layer` |
| Readout training | ridge regression `Wout=YXᵀ(XXᵀ+βI)⁻¹` | `Readout.train_readout` |
| Autonomous prediction | feedback update | `Readout.predict` |
| Genetic algorithm | 5 genes, bounds, integer degree/size | `Optimization.optimize_reservoirs` |
| Prediction-error metrics | NRMSE, NMAE, `J = NMAE / #(NRMSE<ε)` | `Metrics.*` |
| Size–efficiency trade-off | empirical Pareto front + `f(x)=a e^{-bx}+c` | analysis (`find_pareto_netsizeError_64D-KS.jl`) |

> The structural analyses (Laplacian spectra, modularity, EMD, NSGA-II
> multi-objective) live in `original_code/analysis` and consume the `.mat`
> files this simulation produces. Phase 2 of the project will fold those into a
> companion analysis package; this wiki documents the **simulation** half.
