# =============================================================================
# Modularity.jl
# -----------------------------------------------------------------------------
# Community structure, directed modularity, and the regularized connection cost
# of a reservoir's recurrent matrix.
#
# Ports `apply_community_detection`, `compute_modularity`, `average_path_length`,
# `density_of_connections`, and `calculate_regularized_connection_cost` from
# `calculate_modularity_stratifiedSelection.jl`.
#
# Paper §"Community detection and modularity":
#   Communities by label propagation on the weighted directed graph; directed
#   Newman modularity
#       Q = (1/m) Σ_c [ e_c − γ K_c^in K_c^out / m ],  γ = 1,
#   via Graphs.modularity with the weight matrix as distmx.
#
# Paper §"Connection density, path length, and connection cost":
#   density = E / (n(n−1)),  ℓ = mean finite shortest-path length,
#   C = α Σ|A_ij| + β·density + γ·ℓ.
# =============================================================================

module Modularity

using LinearAlgebra
using Graphs: nv, modularity, label_propagation
using SimpleWeightedGraphs: SimpleWeightedDiGraph
using Graphs.Experimental.ShortestPaths: floyd_warshall_shortest_paths

export detect_communities, directed_modularity,
       connection_density, average_path_length, connection_cost,
       NetworkMetrics, network_metrics

# -----------------------------------------------------------------------------
# Community detection
# -----------------------------------------------------------------------------

"""
    detect_communities(A) -> Vector{Int}

Community label per node via label propagation on the weighted **directed**
graph built from adjacency matrix `A`. Port of `apply_community_detection`:
runs `Graphs.label_propagation` on a `SimpleWeightedDiGraph(A)`, then assigns any
node left without a community to a fresh singleton community so the returned
partition vector covers all `nv` nodes with labels `≥ 1`.
"""
function detect_communities(A::AbstractMatrix)
    g = SimpleWeightedDiGraph(A)
    communities = label_propagation(g)        # returns (labels, convergence) or sets

    # Graphs.label_propagation returns a (membership_vector, converged) tuple.
    # Be tolerant: accept either a membership vector or a vector of node-sets.
    labels = communities isa Tuple ? communities[1] : communities

    n = nv(g)
    partition = zeros(Int, n)
    if labels isa AbstractVector{<:Integer} && length(labels) == n
        partition .= labels
    else
        # interpret as an iterable of communities (each a collection of nodes)
        for (i, community) in enumerate(labels)
            for node in community
                partition[node] = i + 1
            end
        end
    end

    # any unassigned node -> its own new community
    for node in findall(==(0), partition)
        partition[node] = maximum(partition) + 1
    end
    return partition
end

# -----------------------------------------------------------------------------
# Directed modularity
# -----------------------------------------------------------------------------

"""
    directed_modularity(A, partition; gamma=1.0) -> Float64

Directed Newman modularity `Q` of the partition under the weighted graph `A`,
computed with `Graphs.modularity` passing the weight matrix through `distmx`
(matching the original `compute_modularity`). `gamma` is the resolution
parameter (`γ = 1` is the traditional definition used in the paper).
"""
function directed_modularity(A::AbstractMatrix, partition::AbstractVector{<:Integer};
                             gamma::Real=1.0)
    g = SimpleWeightedDiGraph(A)
    distmx = Matrix(A)
    return modularity(g, partition; distmx=distmx, γ=gamma)
end

# -----------------------------------------------------------------------------
# Density, path length, connection cost
# -----------------------------------------------------------------------------

"""
    connection_density(A; directed=true) -> Float64

Edge density of `A`. With `directed=true` (the paper's directed normalization),
`density = E / (n(n−1))` where `E` is the number of nonzero entries. With
`directed=false` the original script's symmetric variant `2E / (n(n−1))` is
used.
"""
function connection_density(A::AbstractMatrix; directed::Bool=true)
    n = size(A, 1)
    E = count(!iszero, A)
    return directed ? E / (n * (n - 1)) : 2E / (n * (n - 1))
end

"""
    average_path_length(A) -> Float64

Mean shortest-path length over all ordered node pairs reachable by a finite
path, using Floyd–Warshall on the weighted directed graph. Unreachable pairs
(infinite distance) are skipped from the sum but the original normalization by
`n(n−1)` is kept (port of `average_path_length`).
"""
function average_path_length(A::AbstractMatrix)
    g = SimpleWeightedDiGraph(A)
    n = nv(g)
    sp = floyd_warshall_shortest_paths(g)
    total = 0.0
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        d = sp.dists[i, j]
        isfinite(d) && (total += d)
    end
    return total / (n * (n - 1))
end

"""
    connection_cost(A; alpha=1.0, beta=1.0, gamma=1.0, directed=true) -> Float64

Regularized connection cost
    C = α·Σ|A_ij| + β·density + γ·ℓ,
combining total recurrent weight, connection density, and average path length.
Port of `calculate_regularized_connection_cost` (default weights all one).
"""
function connection_cost(A::AbstractMatrix;
                         alpha::Real=1.0, beta::Real=1.0, gamma::Real=1.0,
                         directed::Bool=true)
    total_weight = sum(abs, A)
    dens = connection_density(A; directed=directed)
    ℓ = average_path_length(A)
    return alpha * total_weight + beta * dens + gamma * ℓ
end

# -----------------------------------------------------------------------------
# Convenience bundle
# -----------------------------------------------------------------------------

"""
    NetworkMetrics

Bundle of the structural measures for one reservoir.

# Fields
- `modularity::Float64`
- `density::Float64`
- `path_length::Float64`
- `cost::Float64`
- `n_communities::Int`
"""
struct NetworkMetrics
    modularity::Float64
    density::Float64
    path_length::Float64
    cost::Float64
    n_communities::Int
end

"""
    network_metrics(A; gamma=1.0, alpha=1.0, beta=1.0, cost_gamma=1.0, directed=true) -> NetworkMetrics

Compute modularity, density, average path length, and connection cost for one
recurrent matrix in a single pass (path length is shared between the cost and
the reported value). `gamma` is the modularity resolution; `cost_gamma` weights
the path-length term of the cost.
"""
function network_metrics(A::AbstractMatrix;
                         gamma::Real=1.0,
                         alpha::Real=1.0, beta::Real=1.0, cost_gamma::Real=1.0,
                         directed::Bool=true)
    partition = detect_communities(A)
    Q = directed_modularity(A, partition; gamma=gamma)
    dens = connection_density(A; directed=directed)
    ℓ = average_path_length(A)
    cost = alpha * sum(abs, A) + beta * dens + cost_gamma * ℓ
    return NetworkMetrics(Q, dens, ℓ, cost, maximum(partition))
end

end # module Modularity
