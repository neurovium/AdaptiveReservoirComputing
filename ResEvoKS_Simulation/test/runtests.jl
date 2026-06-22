# =============================================================================
# test/runtests.jl
# -----------------------------------------------------------------------------
# Unit and integration tests for the ResEvoKS_Simulation simulation package.
#
# Run with:
#   julia --project=. -e 'using Pkg; Pkg.test()'
# or
#   julia --project=. test/runtests.jl
# =============================================================================

using Test
using ResEvoKS_Simulation
using ResEvoKS_Simulation.Reservoir: StableRNG, spectral_radius
using ResEvoKS_Simulation.Readout: square_even_indices!
using LinearAlgebra
using SparseArrays
using Statistics
using MAT

@testset "ResEvoKS_Simulation" begin

    # ---------------------------------------------------------------------
    @testset "KS solver (ETDRK4)" begin
        p = KSModelParams(N=64, d=22.0, tau=0.25, nstep=5_000)
        ic = random_initial_condition(p.N; rng=StableRNG(1))
        u = solve_ks(ic, p)
        @test size(u) == (64, 5_000)
        @test all(isfinite, u)                 # robust: no blow-up
        @test maximum(abs, u) < 10             # bounded KS amplitude (~3)
        @test maximum(abs, u) > 0.5            # nontrivial dynamics developed

        # Reproducibility: same IC + params ⇒ identical trajectory.
        u2 = solve_ks(ic, p)
        @test u == u2

        # Stability over a longer horizon (the historical failure mode).
        plong = KSModelParams(N=64, d=22.0, tau=0.25, nstep=20_000)
        ulong = solve_ks(random_initial_condition(64; rng=StableRNG(7)), plong)
        @test all(isfinite, ulong)
    end

    # ---------------------------------------------------------------------
    @testset "Reservoir construction" begin
        rng = StableRNG(42)
        A, C = generate_reservoir(500, 6; rng=rng)
        @test size(A) == (500, 500)
        @test issparse(A)
        @test all(A.nzval .> 0)                # nonnegative weights
        @test C == (A .> 0)
        # expected density ≈ degree/size
        @test isapprox(nnz(A) / 500^2, 6 / 500; rtol=0.25)

        # spectral-radius scaling
        Ascaled = build_reservoir(512, 6, 0.9; rng=StableRNG(3))
        @test isapprox(spectral_radius(Ascaled), 0.9; rtol=1e-2)

        # input weights: disjoint blocks, range [-sigma, sigma]
        win = generate_input_weights(512, 64, 0.5; rng=StableRNG(9))
        @test size(win) == (512, 64)
        q = 512 ÷ 64
        # column i is nonzero only in its own block of q rows
        for i in 1:64
            block = (i-1)*q+1 : i*q
            @test all(win[setdiff(1:512, block), i] .== 0)
            @test all(abs.(win[block, i]) .<= 0.5 + 1e-12)
        end
    end

    # ---------------------------------------------------------------------
    @testset "Readout feature map" begin
        X = reshape(collect(1.0:8.0), 4, 2)    # 4 × 2
        Y = copy(X)
        square_even_indices!(Y)
        # even rows (2,4) squared; odd rows (1,3) unchanged
        @test Y[1, :] == X[1, :]
        @test Y[3, :] == X[3, :]
        @test Y[2, :] == X[2, :] .^ 2
        @test Y[4, :] == X[4, :] .^ 2

        v = collect(1.0:5.0)
        square_even_indices!(v)
        @test v == [1.0, 4.0, 3.0, 16.0, 5.0]
    end

    # ---------------------------------------------------------------------
    @testset "Error metrics & fitness" begin
        # Perfect prediction ⇒ NRMSE = 0, NMAE = 0.
        truth = randn(StableRNG(2), 8, 100)
        err0 = compute_error(truth, truth)
        @test all(err0.NRMSE .== 0)
        @test err0.NMAE == 0
        # all channels below threshold ⇒ J = NMAE / K = 0
        @test composite_fitness(err0; threshold=0.05) == 0.0

        # No channel below threshold ⇒ J = Inf (zero-denominator rule).
        bad = truth .+ 100.0
        errb = compute_error(bad, truth)
        @test composite_fitness(errb; threshold=0.05) == Inf

        # Explicit zero-denominator rule via a hand-built error.
        e_none = ResEvoKS_Simulation.Metrics.PredictionError(fill(1.0, 8), 0.3)  # all NRMSE ≥ ε
        @test composite_fitness(e_none; threshold=0.05) == Inf

        # Finite J when some channels pass: J = NMAE / (#passing).
        e_some = ResEvoKS_Simulation.Metrics.PredictionError([0.01, 0.01, 1.0, 1.0], 0.2)
        @test composite_fitness(e_some; threshold=0.05) == 0.2 / 2
        # never NaN, always finite-or-Inf
        @test isfinite(composite_fitness(e_some))

        # Diverged rollout: NaN/Inf NMAE even with a passing channel ⇒ Inf, not NaN.
        e_nan = ResEvoKS_Simulation.Metrics.PredictionError([0.01, NaN, NaN, NaN], NaN)
        @test composite_fitness(e_nan; threshold=0.05) == Inf
        e_inf = ResEvoKS_Simulation.Metrics.PredictionError([0.01, 1.0], Inf)
        @test composite_fitness(e_inf; threshold=0.05) == Inf

        # NRMSE matches the textbook formula on a simple case.
        est = truth .+ 0.1 .* randn(StableRNG(5), 8, 100)
        e = compute_error(est, truth)
        for k in 1:8
            expected = sqrt(mean((est[k, :] .- truth[k, :]) .^ 2) / var(truth[k, :]))
            @test isapprox(e.NRMSE[k], expected; rtol=1e-10)
        end
    end

    # ---------------------------------------------------------------------
    @testset "Genome decode" begin
        rp = decode_genome([0.9, 6.4, 1000.0, 0.5, 1.5e-4], 64)
        @test rp.radius == 0.9
        @test rp.degree == 6                  # rounded
        @test rp.N == (1000 ÷ 64) * 64        # snapped to multiple of 64 = 960
        @test rp.N % 64 == 0
        @test rp.sigma == 0.5
        @test rp.beta == 1.5e-4
        @test rp.num_inputs == 64
    end

    # ---------------------------------------------------------------------
    @testset "IO contract (.mat round-trip)" begin
        rp = ResEvoKS_Simulation.Reservoir.ReservoirParams(num_inputs=64, radius=0.873,
                degree=6, N=2048, sigma=0.412, beta=1e-4)
        # filename pattern <gen>_<N>_<deg>_<rad>_<sig> with %f (6 decimals)
        @test individual_filename(12, rp) == "12_2048_6_0.873000_0.412000"

        tmp = mktempdir()
        A = sprand(StableRNG(1), 100, 100, 0.05)
        win = randn(StableRNG(2), 100, 64)
        wout = randn(StableRNG(3), 64, 100)
        err = ResEvoKS_Simulation.Metrics.PredictionError(rand(StableRNG(4), 64), 0.123)
        rp2 = ResEvoKS_Simulation.Reservoir.ReservoirParams(num_inputs=64, radius=0.5,
                degree=4, N=100, sigma=0.3, beta=1e-4)
        # default :mat
        p = save_individual(tmp, 3, A, win, wout, rp2, err, 0.456)
        @test endswith(p, ".mat")
        d = matread(p)
        @test Set(keys(d)) == Set(["w_in", "w_out", "A", "resparams", "err", "J"])
        @test d["J"] == 0.456
        @test d["resparams"]["N"] == 100.0
        @test d["resparams"]["degree"] == 4.0
        @test length(d["err"]["NRMSE"]) == 64
        @test d["err"]["NMAE"] == 0.123
        @test size(d["w_out"]) == (64, 100)

        # load_individual on the .mat reconstructs the structs
        li = load_individual(p)
        @test li.resparams isa ResEvoKS_Simulation.Reservoir.ReservoirParams
        @test li.resparams.N == 100
        @test li.err isa ResEvoKS_Simulation.Metrics.PredictionError
        @test li.err.NMAE == 0.123
        @test li.J == 0.456

        # :jld2 — native Julia round-trip preserves struct + sparsity exactly
        pj = save_individual(tmp, 7, A, win, wout, rp2, err, 0.789; format=:jld2)
        @test endswith(pj, ".jld2")
        lj = load_individual(pj)
        @test lj.resparams == rp2                  # exact struct equality
        @test lj.err.NRMSE == err.NRMSE
        @test lj.J == 0.789
        @test issparse(lj.A) && lj.A == A

        # :both — writes both files; load each
        pb = save_individual(tmp, 9, A, win, wout, rp2, err, 0.5; format=:both)
        @test endswith(pb, ".mat")                 # :both returns the .mat path
        base = first(splitext(pb))
        @test isfile(base * ".mat") && isfile(base * ".jld2")
        @test load_individual(base * ".jld2").J == 0.5
        @test load_individual(base * ".mat").J == 0.5

        # invalid format errors
        @test_throws ArgumentError save_individual(tmp, 1, A, win, wout, rp2, err, 0.1; format=:csv)

        # dataset round-trip (.jld2 keeps ModelParams as a NamedTuple)
        model = KSModelParams(N=64, d=22.0, tau=0.25, nstep=10)
        dpath = save_dataset(tmp, A, model; format=:both)
        @test isfile(first(splitext(dpath)) * ".jld2")
        ld = load_dataset(first(splitext(dpath)) * ".jld2")
        @test ld.ModelParams.N == 64
    end

    # ---------------------------------------------------------------------
    @testset "Single individual end-to-end" begin
        p = KSModelParams(N=64, d=22.0, tau=0.25, nstep=3_000)
        data = solve_ks(random_initial_condition(64; rng=StableRNG(11)), p)
        dp = DataParams(train_length=2_000, predict_length=400)
        res = evaluate_individual([0.9, 6.0, 384.0, 0.5, 1e-4], data, dp;
                                  rng=StableRNG(13))
        @test res.resparams.N == 384
        @test size(res.w_out) == (64, 384)
        @test size(res.w_in) == (384, 64)
        @test length(res.err.NRMSE) == 64
        @test isfinite(res.err.NMAE)
        @test res.J >= 0                      # finite-or-Inf, never negative
    end

    # ---------------------------------------------------------------------
    @testset "GA driver (tiny run)" begin
        p = KSModelParams(N=64, d=22.0, tau=0.25, nstep=3_500)
        data = solve_ks(random_initial_condition(64; rng=StableRNG(21)), p)
        dp = DataParams(train_length=2_200, predict_length=300)
        tmp = mktempdir()
        sp = make_run_dirs(tmp, 1)
        settings = GASettings(population_size=6, max_generations=2,
                    lb=[0.1, 2.0, 128.0, 0.1, 1e-4],
                    ub=[1.0, 6.0, 640.0, 1.0, 2e-4], seed=99)
        result = optimize_reservoirs(data, dp, sp; settings=settings, verbose=false)

        @test length(result.history) == 2
        @test [h.gen for h in result.history] == [0, 1]
        @test size(result.final_population) == (5, 6)

        # per-generation timing + counts are recorded in the history
        @test all(h -> haskey(h, :seconds) && h.seconds >= 0, result.history)
        @test all(h -> haskey(h, :n_failed) && h.n_failed >= 0, result.history)

        # a timestamped log file was written to the run's Log/ directory
        logf = joinpath(sp.codedir, "esn_log.txt")
        @test isfile(logf)
        logtxt = read(logf, String)
        @test occursin("GA run started", logtxt)
        @test occursin("GA finished", logtxt)
        @test occursin(r"gen\s+0/1", logtxt)        # per-generation line present
        @test occursin(r"eval=", logtxt)            # timing present

        # Each saved generation writes one .mat per *successfully built*
        # individual. Tiny test reservoirs (size as small as 128, degree 2) can
        # occasionally fail to build and are skipped by design, so we allow up
        # to 2×6 files and require at least most of them.
        files = filter(f -> endswith(f, ".mat") && f != "data.mat", readdir(sp.matdir))
        @test 2 * 6 - 2 <= length(files) <= 2 * 6
        @test all(f -> occursin(r"^\d+_\d+_\d+_\d+\.\d{6}_\d+\.\d{6}\.mat$", f), files)
        # generations present are 0 and 1
        gens = unique(parse.(Int, first.(split.(files, "_"))))
        @test sort(gens) == [0, 1]

        # reproducibility: same seed ⇒ same best genome
        result2 = optimize_reservoirs(data, dp, make_run_dirs(mktempdir(), 1);
                        settings=settings, verbose=false, save_to_disk=false)
        @test result.best_genome == result2.best_genome
    end

end
