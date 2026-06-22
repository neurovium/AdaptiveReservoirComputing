# =============================================================================
# runtests.jl  —  ResEvoKS_Analysis test suite
# -----------------------------------------------------------------------------
# Synthetic-fixture unit tests for the analysis modules. We fabricate a small
# results directory matching the simulation's on-disk contract
# (<gen>_<N>_<degree>_<radius>_<sigma>.mat with A, w_in, w_out, resparams, err, J)
# and exercise every analysis through it.
# =============================================================================

using Test
using ResEvoKS_Analysis
using MAT
using SparseArrays
using LinearAlgebra
using StableRNGs
using Statistics

# -----------------------------------------------------------------------------
# Fixture builder: a tiny results/RUN1/matfiles tree
# -----------------------------------------------------------------------------

"Build one synthetic individual artifact and write it as a .mat file."
function write_individual(matdir, gen, N, degree, radius, sigma, J;
                          num_inputs=4, rng=StableRNG(1))
    # nonnegative sparse recurrent matrix scaled to spectral radius `radius`
    A = sprand(rng, Float64, N, N, degree / N)
    if nnz(A) > 0
        λ = maximum(abs, eigvals(Matrix(A)))
        λ > 0 && (A .*= radius / λ)
    end
    w_in = zeros(N, num_inputs)
    q = N ÷ num_inputs
    for i in 1:num_inputs
        w_in[(i-1)*q+1:i*q, i] = sigma .* (-1 .+ 2 .* rand(rng, q))
    end
    w_out = randn(rng, num_inputs, N)
    nrmse = clamp.(0.04 .+ 0.02 .* randn(rng, num_inputs), 0.0, 1.0)
    resparams = Dict("N" => Float64(N), "radius" => radius, "degree" => Float64(degree),
                     "sigma" => sigma, "beta" => 1.5e-4, "num_inputs" => Float64(num_inputs))
    err = Dict("NRMSE" => nrmse, "NMAE" => 0.5 * J)  # arbitrary but consistent
    fname = string(gen, "_", N, "_", degree, "_", radius, "_", sigma, ".mat")
    matwrite(joinpath(matdir, fname),
             Dict("A" => A, "w_in" => w_in, "w_out" => w_out,
                  "resparams" => resparams, "err" => err, "J" => J))
    return fname
end

"Create a synthetic run with several generations and return its root dir."
function build_fixture(root)
    matdir = joinpath(root, "RUN1", "matfiles")
    mkpath(matdir)
    rng = StableRNG(20240101)
    # 3 generations, sizes and errors trend down with generation
    for (gi, gen) in enumerate([0, 5, 10])
        for k in 1:12
            N = 16 + 4 * ((k % 4))                 # 16..28, multiple of 4
            degree = 4
            radius = 0.6 + 0.05 * (k % 3)
            sigma = 0.5
            J = 2.0 / gi + 0.1 * k                 # later gens lower J
            write_individual(matdir, gen, N, degree, radius, sigma, J;
                             num_inputs=4, rng=rng)
        end
    end
    # also drop a dataset file that must be excluded
    matwrite(joinpath(matdir, "data.mat"), Dict("dummy" => 1))
    return root
end

# -----------------------------------------------------------------------------
@testset "ResEvoKS_Analysis" begin

mktempdir() do tmp
    root = build_fixture(tmp)
    matdir = run_matdir(root, 1)

    @testset "DataAccess" begin
        files = list_individual_files(matdir)
        @test !("data.mat" in files)
        @test length(files) == 36
        @test parse_generation("10_28_4_0.6_0.5.mat") == 10
        @test available_generations(matdir) == [0, 5, 10]
        @test length(files_for_generation(matdir, 5)) == 12

        rec = load_record(matdir, files[1])
        @test rec.N >= 16
        @test isfinite(rec.J)
        @test length(rec.NRMSE) == 4
        A = load_adjacency(matdir, files[1])
        @test size(A, 1) == size(A, 2)
        J, N = read_J_N(matdir, files[1])
        @test J == rec.J && N == rec.N
        Js, Ns = collect_J_N(matdir, files)
        @test length(Js) == length(files)
    end

    @testset "Sampling" begin
        gens = available_generations(matdir)
        # with our small fixture no generation has >=299, so last gen is fallback
        sel = select_generations(gens, matdir; n_select=10, min_individuals=299)
        @test sel[1] == 0
        @test sel[end] == 10
        @test issorted(sel)

        files = files_for_generation(matdir, 0)
        J = [read_J_N(matdir, f)[1] for f in files]
        s = stratified_sample(files, J; n_per_quartile=2, rng=StableRNG(7))
        @test length(s) <= 8
        @test allunique(s)
        # reproducible with same seed
        s2 = stratified_sample(files, J; n_per_quartile=2, rng=StableRNG(7))
        @test s == s2

        alls = stratified_sample_run(matdir, sel; n_per_quartile=3, rng=StableRNG(7))
        @test !isempty(alls)
    end

    @testset "Pareto" begin
        pts = [1.0 5.0; 2.0 4.0; 3.0 4.5; 2.0 6.0; 4.0 1.0]
        mask, frontier = pareto_frontier(pts)
        # (2,4) and (4,1) and (1,5) are non-dominated; (3,4.5) and (2,6) dominated
        @test mask[2] && mask[5]
        @test !mask[3] && !mask[4]
        @test issorted(frontier[:, 1])

        # exponential fit recovers a known curve
        x = collect(1.0:50.0)
        y = 3.0 .* exp.(-0.1 .* x) .+ 0.5
        fit = fit_exponential(x, y)
        @test isapprox(fit.a, 3.0; atol=1e-3)
        @test isapprox(fit.b, 0.1; atol=1e-3)
        @test isapprox(fit.c, 0.5; atol=1e-3)
        @test fit.r_squared > 0.999
    end

    @testset "Spectral" begin
        A = load_adjacency(matdir, files_for_generation(matdir, 0)[1])
        L = rw_laplacian(A)
        @test size(L) == size(A)
        # row-stochastic D^-1 A => each L row sums to ~0 (for nonzero-degree rows)
        ev = laplacian_eigenvalues(A; nev=:all)
        @test all(isfinite, ev)
        @test issorted(ev)

        grid, Γ = smoothed_density(ev; sigma=0.02, bins=0:0.01:2)
        @test isapprox(sum(Γ), 1.0; atol=1e-8)
        @test length(grid) == length(Γ)
        c = spectral_centroid(grid, Γ)
        @test 0 <= c <= 2

        evlist = [laplacian_eigenvalues(load_adjacency(matdir, f); nev=:all)
                  for f in files_for_generation(matdir, 0)[1:4]]
        g2, M = spectral_density_grid(evlist; bins=0:0.01:2)
        @test size(M, 2) == 4
    end

    @testset "Embedding" begin
        evlist = [laplacian_eigenvalues(load_adjacency(matdir, f); nev=:all)
                  for f in list_individual_files(matdir)[1:16]]
        S = spectra_matrix(evlist, 30)
        @test size(S) == (16, 30)
        # interpolated spectra are monotonic non-decreasing (allow ulp noise)
        @test all(r -> all(>=(-1e-9), diff(r)), eachrow(S))

        pca = spectra_pca(S; maxoutdim=3)
        @test size(pca.scores, 1) == 3
        @test size(pca.scores, 2) == 16
        @test 0 <= pca.cumulative[end] <= 1 + 1e-8

        clus = cluster_spectra(S; max_clusters=5, rng=StableRNG(3))
        @test length(clus.assignments) == 16
        @test 2 <= clus.k <= 5
    end

    @testset "SpectralEMD" begin
        e1 = collect(0.0:0.1:1.0)
        e2 = e1 .+ 0.5
        d = spectral_emd(e1, e2)
        @test d > 0
        @test isapprox(spectral_emd(e1, e1), 0.0; atol=1e-6)

        evlist = [laplacian_eigenvalues(load_adjacency(matdir, f); nev=:all)
                  for f in files_for_generation(matdir, 0)[1:4]]
        D = pairwise_emd(evlist; show_progress=false)
        @test size(D) == (4, 4)
        @test issymmetric(D)
        @test all(==(0), diag(D))
        ref = reduce(vcat, evlist)
        r = emd_to_reference(evlist, ref; show_progress=false)
        @test length(r) == 4
    end

    @testset "Modularity" begin
        A = load_adjacency(matdir, files_for_generation(matdir, 0)[1])
        part = detect_communities(A)
        @test length(part) == size(A, 1)
        @test all(>=(1), part)

        Q = directed_modularity(A, part)
        @test isfinite(Q)
        @test -1 <= Q <= 1

        dens = connection_density(A)
        @test 0 <= dens <= 1
        ℓ = average_path_length(A)
        @test ℓ >= 0
        c = connection_cost(A)
        @test c > 0

        m = network_metrics(A)
        # label propagation is stochastic, so network_metrics may find a
        # different partition than the standalone call above; only assert the
        # value is a valid modularity, not bitwise equality with Q.
        @test isfinite(m.modularity) && -1 <= m.modularity <= 1
        @test m.density == dens
        @test m.n_communities >= 1
    end

    @testset "MultiObjective" begin
        @test normalize_metric([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
        @test normalize_metric([5.0, 5.0]) == [0.0, 0.0]

        nm = normalize_metrics(modularity=[0.1, 0.5, 0.9],
                               connection_cost=[10.0, 20.0, 30.0],
                               performance=[2.0, 1.0, 0.5],
                               generation=[0, 5, 10])
        @test all(0 .<= nm.performance .<= 1)
        o = composite_objectives((0.5, 0.5, 0.5, 0.5))
        @test length(o) == 4
        @test all(isfinite, o)

        optimized = run_nsga2(N=60, p_cr=0.85, p_m=0.5)
        @test size(optimized, 2) == 4
        @test all(0 .<= optimized .<= 1)
        matches = closest_observed(optimized, nm)
        @test all(1 .<= matches .<= 3)
    end

    @testset "ReferenceGraphs" begin
        er = er_spectrum(60, 0.1; rng=StableRNG(1))
        @test length(er) == 60
        @test all(isfinite, er)
        ba = ba_spectrum(60, 2; rng=StableRNG(1))
        @test length(ba) == 60
        ws = ws_spectrum(60, 6, 0.1; rng=StableRNG(1))
        @test length(ws) == 60
        sbm = sbm_spectrum([30, 30], [0.8 0.05; 0.05 0.8]; rng=StableRNG(1))
        @test length(sbm) == 60
        bundle = reference_spectra(n=60; rng=StableRNG(1))
        @test Set(keys(bundle)) == Set([:er, :ba, :ws, :sbm])
    end

    @testset "ErrorStats" begin
        gens = [0, 5, 10]
        errs = collect_generation_errors(matdir, gens)
        @test length(errs[0]) == 12
        lg = log_error_distribution(errs[0])
        @test all(isfinite, lg)

        stats = generation_error_stats(matdir, gens)
        @test length(stats) == 3
        @test stats[1].generation == 0
        @test stats[1].n == 12
        # later generations should have lower median error in our fixture
        @test stats[end].median < stats[1].median
    end
end

end # testset
