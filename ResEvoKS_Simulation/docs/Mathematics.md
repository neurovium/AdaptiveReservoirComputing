# Mathematics

This page presents the mathematics of the **simulation** half of the paper in a
digestible form, and links each equation to the function that implements it.
Notation follows the paper's Methods section.

---

## Kuramoto–Sivashinsky solver

### The equation

The one-dimensional Kuramoto–Sivashinsky (KS) equation on a periodic domain of
length `d` (the paper's `L`) is

```
∂u/∂t + u ∂u/∂x + ∂²u/∂x² + ∂⁴u/∂x⁴ = 0,        u(x, t),  x ∈ [0, d).
```

It combines nonlinear advection (`u u_x`), a *long-wave instability* (`u_xx`,
energy-injecting), and *short-wave dissipation* (`u_xxxx`). The balance produces
sustained spatiotemporal chaos for `d = 22`, the canonical value used here.

### Spectral (Fourier) discretization

With `N` grid points and the discrete wavenumbers

```
k = (2π/d) · [0, 1, …, N/2−1, 0, −N/2+1, …, −1]ᵀ,
```

differentiation becomes multiplication: `∂ₓ → ik`. The **linear part** is the
diagonal Fourier multiplier

```
ℒ_k = k² − k⁴
```

(positive for small `k` → growth; strongly negative for large `k` → damping).
The **nonlinear part** `−½ ∂ₓ(u²)` is computed pseudo-spectrally as
`N(û) = −½ ik · 𝔉{ (𝔉⁻¹û)² }`.

### Time integration: ETDRK4

The linear part is stiff, so the integrator treats it exactly via the matrix
exponential and the nonlinear part explicitly. This is the **fourth-order
exponential time-differencing Runge–Kutta** method (Cox & Matthews 2002), with
the coefficients evaluated by the contour-integral trick of Kassam & Trefethen
(2005) to avoid catastrophic cancellation:

```
E  = exp(h ℒ),     E2 = exp(h ℒ / 2),
Q, f1, f2, f3 = h · mean over M=16 contour points of the ETD ϕ-functions.
```

One ETDRK4 step advances `v = û` through four nonlinear evaluations
(`Nv, Na, Nb, Nc`) and the update

```
v⁺ = E·v + Nv·f1 + 2(Na+Nb)·f2 + Nc·f3.
```

**Implemented in:** [`KuramotoSivashinsky.solve_ks`](../src/KuramotoSivashinsky.jl).

### A note on numerical stability (real vs. complex FFT)

KS is sensitive: `u` is real, so its spectrum must be Hermitian-symmetric, but a
generic *complex* FFT lets floating-point roundoff inject a tiny anti-Hermitian
component that the unstable modes amplify until the run blows up (around
`t ≈ 360`). The original MATLAB/Python code used the complex FFT. The
particular numerics of `scipy.fft` survives the instability. This port uses the **real FFT**
(`rfft`/`irfft`), which represents the field by its nonnegative-wavenumber half
and therefore *cannot* develop the spurious mode — the identical ETDRK4 scheme,
but stable on any backend. Since KS is chaotic, no two implementations match
trajectory-by-trajectory anyway; the paper's claims are statistical/structural.

---

## Reservoir construction

### Recurrent matrix

A reservoir is a sparse, directed, weighted recurrent network `A ∈ ℝ^{n_r×n_r}`.
The target average degree `d` sets the connection probability (sparsity)

```
p = d / n_r,
```

and nonzero weights are drawn i.i.d. uniform on `(0, 1)`. The matrix is then
rescaled to a target **spectral radius** `ρ_target`:

```
A ← ρ_target · A₀ / λ_max,        λ_max = maxₗ |λₗ(A₀)|,
```

so that `ρ(A) = ρ_target`. The spectral radius controls the reservoir's memory
and proximity to the "edge of chaos".

**Implemented in:** [`Reservoir.generate_reservoir`](../src/Reservoir.jl),
[`Reservoir.scale_spectral_radius!`](../src/Reservoir.jl),
[`Reservoir.build_reservoir`](../src/Reservoir.jl). The dominant eigenvalue uses
the thread-safe pure-Julia `KrylovKit.eigsolve` (see
[Running & Reproducibility](Running-and-Reproducibility.md#why-krylovkit-not-arpack)).

### Input matrix (block design)

The reservoir size is an integer multiple of the input dimension,
`n_r = q · n_u`, so each input channel `i` drives its own disjoint block of
`q = n_r/n_u` reservoir nodes. Within the block, weights are uniform on
`[−σ, σ]`; outside it they are zero. `σ` is the **input scaling**.

**Implemented in:** [`Reservoir.generate_input_weights`](../src/Reservoir.jl).

---

## Reservoir dynamics and readout

### State update (teacher forcing)

Driven by the true input `u(t)`, the reservoir state evolves as

```
x(t) = f( A x(t−1) + W_in u(t) ),        x(0) = 0,
```

with `f = tanh` (elementwise). States are collected over the training window.

**Implemented in:** [`Readout.reservoir_layer`](../src/Readout.jl).

### Bilinear feature map

Before the readout, every **even-indexed** reservoir unit is squared:

```
x_aug = [x₁, x₂², x₃, x₄², …].
```

This breaks the `u → −u` symmetry of the readout (the Pathak et al. trick) and
is essential for KS prediction.

**Implemented in:** [`Readout.square_even_indices!`](../src/Readout.jl).

### Ridge-regression readout

The only trained component is the linear readout `W_out`. With the augmented
state matrix `X_aug` (columns over the training window) and targets `Y` (the KS
field), the Tikhonov-regularized solution is

```
W_out = Y X_augᵀ ( X_aug X_augᵀ + β I )⁻¹,
```

where `β` is the ridge regularization. The output is `y(t) = W_out x_aug(t)`.

**Implemented in:** [`Readout.train_readout`](../src/Readout.jl). (For the
well-conditioned `β>0` system we solve the normal equations directly rather than
forming a pseudoinverse — numerically equivalent, much cheaper.)

### Autonomous (closed-loop) prediction

To measure forecast skill, the trained reservoir runs **without** the true
input: its own output is fed back.

```
y(t)   = W_out x_aug(t)
x(t+1) = f( A x(t) + W_in y(t) ).
```

Errors over this horizon quantify how long the reservoir tracks the chaos.

**Implemented in:** [`Readout.predict`](../src/Readout.jl).

---

## Prediction-error metrics and fitness

### NRMSE (per channel)

For each output channel `k` over a horizon of `T` samples,

```
NRMSE_k = sqrt( (1/T) Σₜ (y_k(t) − ŷ_k(t))² ) / σ_{ŷ_k},
```

normalized by the standard deviation of the *true* channel.

### NMAE (scalar)

A single mean-absolute error after a min–max normalization that uses the min/max
of the *predicted* output for both signals (this exact convention is reproduced from `compute_error.m` because it defines `J`):

```
ã = (y_target − min ŷ) / max(y_target − min ŷ),
b̃ = (ŷ        − min ŷ) / max(y_target − min ŷ),
NMAE = (1/KT) Σ_{k,t} | b̃_k(t) − ã_k(t) |.
```

### Composite fitness `J`

```
J = NMAE / Σ_k 𝟙[ NRMSE_k < ε ],        ε = 0.05.
```

The denominator counts how many spatial channels are predicted below the error
threshold; if none are, `J = ∞` (worst fitness). Lower `J` is better. The GA
minimizes `J`. This rewards reservoirs that are *both* accurate (low NMAE) *and*
broad (many channels under threshold) — appropriate for a spatially extended
system.

**Implemented in:** [`Metrics.compute_error`](../src/Metrics.jl),
[`Metrics.composite_fitness`](../src/Metrics.jl).

---

## Genetic algorithm

The GA optimizes the five construction hyperparameters

```
genome = [ ρ (radius), d (degree), n_r (size), σ (input scaling), β (ridge) ],
```

over the box

| gene | ρ | d | n_r | σ | β |
|------|---|---|-----|---|---|
| lower | 0.1 | 2 | 300 | 0.1 | 1e-4 |
| upper | 1.0 | 10 | 3000 | 1.0 | 2e-4 |

with `d` and `n_r` constrained to integers (and `n_r` snapped to a multiple of
`n_u`). Each generation: evaluate the population (in parallel), select by
tournament with elitism, recombine by blend (BLX-α) crossover, and mutate by
Gaussian creep with occasional uniform reset — the standard real-coded GA stages
described in the paper. Lower `J` is fitter.

**Implemented in:** [`Optimization.optimize_reservoirs`](../src/Optimization.jl).

---

## Size–efficiency frontier (post-hoc analysis)

Each evaluated reservoir is a point `(n_r, log J)`. The empirical **Pareto
frontier** keeps points that no other dominates in both size and error. Its
shape is summarized by a nonlinear least-squares exponential fit

```
f(x) = a e^{−b x} + c,
```

capturing the diminishing-return relationship between reservoir size and
attainable error. This analysis is performed on the saved `.mat` files by
`original_code/analysis/KS_analysis/find_pareto_netsizeError_64D-KS.jl`.

---

## References

- Y. Kuramoto (1978); G. Sivashinsky (1977) — the KS equation.
- S. M. Cox & P. C. Matthews (2002), *Exponential time differencing for stiff
  systems* — ETDRK4.
- A.-K. Kassam & L. N. Trefethen (2005), *Fourth-order time-stepping for stiff
  PDEs* — the contour-integral ETD coefficients.
- J. Pathak et al. (2018), *Model-free prediction of large spatiotemporally
  chaotic systems* — reservoir prediction of KS with the bilinear readout.
- N. Dehghani (2026), *Evolutionary Optimization Reveals Structural Constraints on Reservoir Architecture for Spatiotemporal Chaos* — evolutionary optimization of reservoir prediction of KS.  
