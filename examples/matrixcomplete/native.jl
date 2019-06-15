#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see https://www.cvxpy.org/examples/dgp/pf_matrixcomplete.html
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

function matrixcomplete(
    m::Int,
    n::Int;
    num_known::Int = -1,
    known_rows::Vector{Int} = Int[],
    known_cols::Vector{Int} = Int[],
    known_vals::Vector{Float64} = Float64[],
    use_3dim::Bool = false,
    )
    @assert m >= n

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
    mat_to_vec_idx(i, j) = (j - 1) * m + i

    # a little more confusing
    # known_idxs = mat_to_vec_idx.(known_rows, known_cols)
    # p = sortperm(known_idxs)
    # unknown_idxs = setdiff(1:(m * n), known_idxs)
    # is_known = fill(false, m * n)
    # is_known[known_idxs] .= true
    # G1[unique(unknown_idxs[p]), 1:num_unknown] .= -1
    # h1[known_idxs[p]] .= known_vals[p] # if an index is repeated, last value will be chosen

    is_known = fill(false, m * n)
    h1 = zeros(m * n)
    for (k, (i, j)) in enumerate(zip(known_rows, known_cols))
        known_idx = mat_to_vec_idx(i, j)
        h1[known_idx] = known_vals[k]
        is_known[known_idx] = true
    end

    num_known = sum(is_known) # if randomly generated, some indices may repeat
    num_unknown = m * n - num_known
    c = zeros(1 + num_unknown)
    c[1] = 1
    b = Float64[]

    # epinormspectral cone- get vec(X) in G and h
    G1 = zeros(m * n, num_unknown)
    total_idx = 1
    unknown_idx = 1
    for j in 1:n, i in 1:m
        if !is_known[mat_to_vec_idx(i, j)]
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

    spectral_dim = m * n + 1

    if !use_3dim
        # hypogeomean for values to be filled
        G2 = zeros(num_unknown + 1, num_unknown + 1)
        total_idx = 1
        unknown_idx = 1
        for j in 1:n, i in 1:m
            if !is_known[mat_to_vec_idx(i, j)]
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

        A = zeros(0, 1 + num_unknown)

        push!(cone_idxs, (m * n + 2):(m * n + 2 + num_unknown))
        push!(cones, CO.HypoGeomean{Float64}(ones(num_unknown) / num_unknown))

    else
        # number of 3-dimensional power cones needed is num_unknown - 1, number of new variables is num_unknown - 2
        # first num_unknown columns overlap with G1, column for the epigraph variable of the spectral cone added later
        G2 = zeros(3 * (num_unknown - 1), 2 * num_unknown - 2)
        # first cone is a special case since two of the original variables participate in it
        G2[3, 1] = -1
        G2[2, 2] = -1
        G2[1, num_unknown + 1] = -1
        push!(cones, CO.HypoGeomean{Float64}([0.5, 0.5]))
        push!(cone_idxs, (spectral_dim + 1):(spectral_dim + 3))
        offset = 4
        # loop over new vars
        for i in 1:(num_unknown - 3)
            G2[offset, num_unknown + i + 1] = -1
            G2[offset + 1, num_unknown + i] = -1
            G2[offset + 2, i + 2] = -1
            push!(cones, CO.HypoGeomean{Float64}([(i + 1) / (i + 2), 1 / (i + 2)]))
            push!(cone_idxs, (spectral_dim + 3 * i + 1):(spectral_dim + 3 * (i + 1)))
            offset += 3
        end
        # last row also special becuase hypograph variable is fixed
        G2[offset + 2, num_unknown] = -1
        G2[offset + 1, 2 * num_unknown - 2] = -1
        push!(cones, CO.HypoGeomean{Float64}([(num_unknown - 1) / num_unknown, 1 / num_unknown]))
        push!(cone_idxs, (spectral_dim + 3 * num_unknown - 5):(spectral_dim + 3 * num_unknown - 3))
        h = vcat(h1, zeros(3 * (num_unknown - 2)), [1, 0, 0])

        # G1 needs to be post-padded with columns for 3dim cone vars
        G1 = [G1 zeros(m * n + 1, num_unknown - 2)]
        # G2 needs to be pre-padded with the epigraph variable for the spectral norm cone
        G2 = [zeros(3 * (num_unknown - 1)) G2]
        G = vcat(G1, G2)
        c = vcat(c, zeros(num_unknown - 2))
        A = zeros(0, size(G, 2))
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

matrixcomplete_ex(use_3dim::Bool) = matrixcomplete(
    3,
    3,
    num_known = 5,
    known_rows = [1, 3, 2, 3, 1],
    known_cols = [1, 1, 2, 2, 3],
    known_vals = [1.0, 3.2, 0.8, 5.9, 1.9],
    use_3dim = use_3dim,
    )
matrixcomplete1() = matrixcomplete_ex(false)
matrixcomplete2() = matrixcomplete_ex(true)
matrixcomplete3() = matrixcomplete(6, 5, use_3dim = false)
matrixcomplete4() = matrixcomplete(6, 5, use_3dim = true)

# [
# 1.        0.6911765108350109 1.9
# 1.031601017350596 0.8        2.3506257965881243
# 3.2        5.9        0.5966455412012085
# ]

# [1.         4.63616907 1.9
# 0.49991744 0.8        0.37774148
# 3.2        5.9        1.14221476]

function test_matrixcomplete(instance::Function; options, rseed::Int = 1)
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

test_matrixcomplete_all(; options...) = test_matrixcomplete.([
    matrixcomplete1,
    matrixcomplete2,
    matrixcomplete3,
    matrixcomplete4,
    ], options = options)

test_matrixcomplete(; options...) = test_matrixcomplete.([
    matrixcomplete1,
    matrixcomplete2,
    matrixcomplete3,
    matrixcomplete4,
    ], options = options)
