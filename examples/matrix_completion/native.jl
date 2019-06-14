#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see https://www.cvxpy.org/examples/dgp/pf_matrix_completion.html
modified to use spectral norm

=#

using LinearAlgebra
using SparseArrays
import Random
using Test
import DynamicPolynomials
import Hypatia
const HYP = Hypatia
const MO = HYP.Models
const MU = HYP.ModelUtilities
const CO = HYP.Cones
const SO = HYP.Solvers

function matrix_completion(
    m::Int,
    n::Int;
    num_known::Int = -1,
    known_rows::Vector{Int} = Int[],
    known_cols::Vector{Int} = Int[],
    known_vals::Vector{Float64} = Float64[],
    use_3dim::Bool = false,
    )

    if num_known < 0
        num_known = round(Int, m * n * 0.1)
    end
    if isempty(known_rows)
        known_rows = rand(1:m, num_known)
    end
    if isempty(known_cols)
        known_cols = rand(1:n, num_known)
    end
    if isempty(known_vals)
        known_vals = randn(num_known)
    end
    known_pairs = [(known_rows[i], known_cols[i]) for i in 1:num_known]
    cart_to_single(i, j) = (j - 1) * m + i

    num_unknown = m * n - num_known
    dimx = 1 + num_unknown
    c = zeros(dimx)
    c[1] = 1
    A = zeros(0, dimx)
    b = Float64[]

    # epinormspectral cone- get vec(X) in G and h
    G1 = zeros(m * n, num_unknown)
    h1 = zeros(m * n)

    is_known = fill(false, n * m)
    for (k, (i, j)) in enumerate(zip(known_rows, known_cols))
        known_idx = cart_to_single(i, j)
        h1[known_idx] = known_vals[k]
        is_known[known_idx] = true
    end

    total_idx = 1
    unknown_idx = 1
    for j in 1:n, i in 1:m
        if !is_known[cart_to_single(i, j)]
            G1[total_idx, unknown_idx] = -1
            unknown_idx += 1
        end
        total_idx += 1
    end

    # add first row and column for epigraph variable
    G1 = [-1 zeros(1, num_unknown); zeros(m * n) G1]
    h1 = [0, h1...]

    cones = CO.Cone[CO.EpiNormSpectral{Float64}(n, m)]
    cone_idxs = [1:(m * n + 1)]

    if !use_3dim
        # hypogeomean for values to be filled
        G2 = zeros(num_unknown + 1, num_unknown + 1)
        total_idx = 1
        unknown_idx = 1
        for j in 1:n, i in 1:m
            if !is_known[cart_to_single(i, j)]
                G2[unknown_idx + 1, unknown_idx + 1] = -1
                unknown_idx += 1
            end
            total_idx += 1
        end
        # first component of the vector in the in geomean cone, elements multiply to one
        h2 = [1, zeros(num_unknown)...]
        h = vcat(h1, h2)
        @assert total_idx - 1 == m * n
        @assert unknown_idx - 1 == num_unknown

        G = vcat(G1, G2)

        push!(cone_idxs, (m * n + 2):(m * n + 2 + num_unknown))
        push!(cones, CO.HypoGeomean{Float64}(ones(num_unknown) / num_unknown))
    end

    if use_3dim
        dimx += num_unknown - 2
        # number of 3-dimensional power cones needed is num_unknown - 1
        G3 = zeros(3 * (num_unknown - 1), dimx)
        # first cone is a special case since two of the original variables participate in it
        # TODO offset columns by 1 due to spectral variable
        G3[3, 1] = -1
        G3[2, 2] = -1
        G3[1, num_unknown + 1] = -1
        push!(cones, CO.HypoGeomean{Float64}([0.5, 0.5]))
        push!(cone_idxs, spectral_dim + 1:3)
        offset = 4
        # loop over new vars
        for i in 1:(num_unknown - 3)
            G3[offset, num_unknown + i + 1] = -1
            G3[offset + 1, num_unknown + i] = -1
            G3[offset + 2, i + 2] = -1
            push!(cones, CO.HypoGeomean{Float64}([(i + 1) / (i + 2), 1 / (i + 2)]))
            push!(cone_idxs, spectral_dim + (3 * i + 1):(3 * (i + 1)))
            offset += 3
        end
        # last row also special becuase hypograph variable is fixed
        G3[offset + 2, num_unknown] = -1
        G3[offset + 1, 2 * num_unknown - 2] = -1
        @show G3
        push!(cones, CO.HypoGeomean{Float64}([(num_unknown - 1) / num_unknown, 1 / num_unknown]))
        push!(cone_idxs, spectral_dim + (num_unknown - 5):(num_unknown - 3))
        h = vcat(h1, zeros(3 * (num_unknown - 2)), [1, 0, 0])
        @show h

        G3 = [zeros(3 * (num_unknown - 1)) G3]
        G = vcat([G1 zeros(1, num_unknown - 1)], [G2, zeros(num_unknown - 1)], G3)

    end

    @show A, G, h cone_idxs, cones

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

matrix_completion1() = matrix_completion(
    3,
    3,
    num_known = 5,
    known_rows = [1, 3, 2, 3, 1],
    known_cols = [1, 1, 2, 2, 3],
    known_vals = [1.0, 3.2, 0.8, 5.9, 1.9],
)

# [
# 1.        0.6911765108350109 1.9
# 1.031601017350596 0.8        2.3506257965881243
# 3.2        5.9        0.5966455412012085
# ]

function test_matrix_completion(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    model = MO.PreprocessedLinearModel{Float64}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{Float64}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
    @show r.x
    @test r.status == :Optimal
    return
end

test_matrix_completion_all(; options...) = test_matrix_completion.([
    matrix_completion1,
    ], options = options)

test_matrix_completion(; options...) = test_matrix_completion.([
    matrix_completion1,
    ], options = options)
