# Mathematics

The analyses in digestible form, each tied to the function that implements it.
Notation: a reservoir is a directed weighted graph with `n_r × n_r` recurrent
matrix `A` (after spectral-radius scaling); `A_ij ≥ 0` is the weight from node
`i` to node `j`. `J` is the composite forecast error (lower is better).

---

## 1. Size–efficiency Pareto front  → `Pareto`

Each evaluated reservoir is a point in the size–error plane,

```
x_i = n_{r,i},      y_i = log J_i .
```

The logarithm emphasizes differences among the good reservoirs. A point
`(x_i, y_i)` is **Pareto efficient** (non-dominated) if no other `(x_j, y_j)` has

```
x_j ≤ x_i   and   y_j ≤ y_i ,   with at least one strict inequality.
```

`pareto_frontier(points)` returns the non-dominated mask and the frontier rows
(sorted by size). The frontier is summarized by a nonlinear least-squares fit of

```
f(x) = a · e^{-b x} + c                              (Pareto.exponential_model)
```

via `fit_exponential`, which also reports `R²`. The decay `b` measures how fast
attainable error drops with size; `c` is the asymptotic floor.

---

## 2. Network selection  → `Sampling`

To track structure across evolution at feasible cost (paper §"Network selection
for spectral analysis"):

- **Generations:** the first generation, the last generation with ≥ 299
  individuals, and evenly spaced generations between them
  (`select_generations`, default 10).
- **Within a generation:** rank reservoirs by `J`, split into 4 quartiles,
  sample ≤ 20 per quartile — up to 80 reservoirs per generation
  (`stratified_sample`). This captures elite *and* typical structures.

---

## 3. Random-walk Laplacian spectrum  → `Spectral`

For the directed weighted `A`, the row weight (out-strength) of node `i` is

```
d_i = Σ_j A_ij ,      D = diag(d_1, …, d_{n_r}) .
```

The **random-walk normalized Laplacian** is

```
L_rw = I − D⁻¹ A .                                   (Spectral.rw_laplacian)
```

`D⁻¹A` is row-stochastic (for positive row sums), so the eigenvalues of `L_rw`
lie in the unit disk centered at `1` — a bounded domain comparable across
networks of different size and density. Because `A` is directed and asymmetric,
eigenvalues may be complex; analyses use their **real parts**
(`laplacian_eigenvalues`, ARPACK `eigs`, smallest-magnitude by default, dense
fallback for tiny matrices).

---

## 4. Smoothed eigenvalue density  → `Spectral`

A discrete spectrum `{λ_i}_{i=1}^m` is turned into a smooth curve by Gaussian
kernel smoothing,

```
Γ(x) = (1/m) Σ_i  N(x; λ_i, s²) ,                    (Spectral.smoothed_density)
```

evaluated on a grid and renormalized so it sums to one (bandwidth `s = 0.015` by
default). `spectral_density_grid` stacks many densities as columns;
`spectral_centroid` is the weighted-mean eigenvalue `Σ x Γ(x) / Σ Γ(x)`. Averaging
`Γ` within a generation and stacking generations gives the spectral-evolution
heatmap.

---

## 5. Fixed-length spectra and PCA  → `Embedding`

Reservoirs differ in size, so spectra differ in length. Each spectrum is sorted
and linearly interpolated to a common length `q`,

```
Λ̃_i ∈ ℝ^q ,                                         (Embedding.interpolate_spectrum)
```

and stacked into `S ∈ ℝ^{M×q}` (`spectra_matrix`). Principal-component analysis
of `S` (`spectra_pca`, features = eigenvalue positions) yields low-dimensional
coordinates; coloring by generation shows whether the population follows a
coherent structural trajectory. `cluster_spectra` runs k-means and picks `k` by
the best mean silhouette.

---

## 6. Spectral distance via optimal transport  → `SpectralEMD`

Two spectra are compared as empirical measures with equal mass `1/m` per
eigenvalue. With quadratic ground cost `C_ij = (λ_i⁽¹⁾ − λ_j⁽²⁾)²`, the Earth
Mover's Distance is the optimal-transport cost

```
EMD(Λ⁽¹⁾, Λ⁽²⁾) = min_{γ ∈ Π(a,b)} Σ_ij γ_ij C_ij ,   (SpectralEMD.spectral_emd)
```

solved exactly (`OptimalTransport.emd2` + `Tulip`). This is a squared-2-
Wasserstein-type distance for the 1-D empirical distributions.
`pairwise_emd` builds the full distance matrix; `emd_to_reference` measures each
spectrum's drift from the generation-0 reference.

---

## 7. Community structure and modularity  → `Modularity`

Communities are found by **label propagation** on the weighted directed graph
(`detect_communities`). For a partition `{c}`, the directed Newman modularity is

```
Q = (1/m) Σ_c [ e_c − γ · K_c^in K_c^out / m ] ,     (Modularity.directed_modularity)
```

where `m` is total edge weight, `e_c` the weight of within-community edges, and
`K_c^in`, `K_c^out` the summed in/out degrees of community `c`. The paper uses
`γ = 1`; computed via `Graphs.modularity` with the weight matrix as `distmx`.

---

## 8. Density, path length, connection cost  → `Modularity`

```
density = E / (n_r (n_r − 1))                        (connection_density)
```

with `E` the number of directed nonzero connections (a symmetric `2E/…` variant
is available). The mean shortest-path length over reachable ordered pairs,

```
ℓ = (1/|P|) Σ_{(i,j)∈P} d_ij ,                       (average_path_length)
```

uses Floyd–Warshall on the weighted digraph. The regularized **connection cost**
combines weight, density, and path length,

```
C = α Σ_ij |A_ij| + β · density + γ · ℓ .            (connection_cost)
```

`network_metrics` returns all of these for one reservoir in a single pass.

---

## 9. Multi-objective trade-off (NSGA-II)  → `MultiObjective`

Each metric is min–max normalized to `[0,1]`,

```
x_norm = (x − min x) / (max x − min x) .              (normalize_metric)
```

Four composite objectives over `(modularity, cost, performance, generation)`:

```
O₁ = perf / (1 + gen)        # improvement across generations
O₂ = cost / (1 + mod)        # structural efficiency
O₃ = perf / (1 + mod)        # performance–modularity
O₄ = perf · (1 + cost)       # performance–cost            (composite_objectives)
```

The additive `1+` terms keep `O₁`–`O₃` finite at zero and keep cost positive in
`O₄`. NSGA-II (`run_nsga2`, `N=1000, p_cr=0.85, p_m=0.5`) searches the bounded
`[0,1]⁴` objective space; `closest_observed` matches each Pareto-optimal point to
the nearest real reservoir by Euclidean distance. **This NSGA-II is independent
of the GA that evolved the reservoirs** — it is applied post hoc to the
normalized metrics.

---

## 10. Reference spectral envelope  → `ReferenceGraphs`

Canonical ensembles under the *same* random-walk Laplacian provide a backdrop:
Erdős–Rényi (random), Barabási–Albert (power-law), Watts–Strogatz (small-world),
and the Stochastic Block Model (modular). Comparing the evolved-reservoir spectra
against these places them in an **SBM-like signature class**
(`er_spectrum`, `ba_spectrum`, `ws_spectrum`, `sbm_spectrum`).
