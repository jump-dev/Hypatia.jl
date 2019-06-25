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

THR = Type{<: HYP.HypReal}

function matrixcompletion(
    m::Int,
    n::Int;
    num_known::Int = -1,
    known_rows::Vector{Int} = Int[],
    known_cols::Vector{Int} = Int[],
    known_vals::Vector{Float64} = Float64[],
    use_geomean::Bool = true,
    use_epinorm::Bool = true,
    T::THR = Float64,
    )
    @assert m <= n

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

    mat_to_vec_idx(i::Int, j::Int) = (j - 1) * m + i

    is_known = fill(false, m * n)
    # h for the rows that X (the matrix and not epigraph variable) participates in
    h_norm_x = zeros(m * n)
    for (k, (i, j)) in enumerate(zip(known_rows, known_cols))
        known_idx = mat_to_vec_idx(i, j)
        # if not using the epinorminf cone, indices relate to X'
        h_norm_x[known_idx] = known_vals[k]
        is_known[known_idx] = true
    end

    num_known = sum(is_known) # if randomly generated, some indices may repeat
    num_unknown = m * n - num_known
    c = vcat(1, zeros(num_unknown))
    b = Float64[]

    # epinormspectral cone- get vec(X) in G and h
    if use_epinorm
        G_norm = zeros(m * n, num_unknown)
        total_idx = 1
        unknown_idx = 1
        for j in 1:n, i in 1:m
            if !is_known[total_idx]
                G_norm[total_idx, unknown_idx] = -1
                unknown_idx += 1
            end
            total_idx += 1
        end

        # add first row and column for epigraph variable
        G_norm = [-1 zeros(1, num_unknown); zeros(m * n) G_norm]
        h_norm_x = vcat(0, h_norm_x)
        h_norm = h_norm_x

        cones = CO.Cone[CO.EpiNormSpectral{T}(m, n)]
        cone_idxs = UnitRange{Int}[1:(m * n + 1)]
        cone_offset = m * n + 1
    else
        num_rows = div(m * (m + 1), 2) + m * n + div(n * (n + 1), 2)
        G_norm = zeros(num_rows, num_unknown + 1)
        h_norm = zeros(num_rows)
        # first block epigraph variable * I
        for i in 1:m
            G_norm[sum(1:i), 1] = -1
        end
        offset = div(m * (m + 1), 2)
        # index to count rows in the bottom half of the large to-be-PSD matrix
        idx = 1
        # index only in X
        var_idx = 1
        # index of unknown vars (the x variables in the standard from), can increment it because we are moving row wise in X'
        unknown_idx = 1
        # fill bottom `n` rows
        for i in 1:n
            # X'
            for j in 1:m
                if !is_known[var_idx]
                    G_norm[offset + idx, 1 + unknown_idx] = -rt2
                    unknown_idx += 1
                else
                    h_norm[offset + idx] = h_norm_x[var_idx] * rt2
                end
                idx += 1
                var_idx += 1
            end
            # second block epigraph variable * I
            # skip `i` rows which will be filled with zeros
            idx += i
            G_norm[offset + idx - 1, 1] = -1
        end
        cones = CO.Cone[CO.PosSemidef{T, T}(num_rows)]
        cone_idxs = UnitRange{Int}[1:num_rows]
        cone_offset = num_rows
    end

    if use_geomean
        # hypogeomean for values to be filled
        G_geo = zeros(num_unknown + 1, num_unknown + 1)
        total_idx = 1
        unknown_idx = 1
        for j in 1:n, i in 1:m
            if !is_known[mat_to_vec_idx(i, j)]
                G_geo[unknown_idx + 1, unknown_idx + 1] = -1
                unknown_idx += 1
            end
            total_idx += 1
        end
        # first component of the vector in the in geomean cone, elements multiply to one
        h2 = vcat(1, zeros(num_unknown))
        h = vcat(h_norm, h2)
        @assert total_idx - 1 == m * n
        @assert unknown_idx - 1 == num_unknown

        A = zeros(0, 1 + num_unknown)
        push!(cone_idxs, (cone_offset + 1):(cone_offset + num_unknown + 1))
        push!(cones, CO.HypoGeomean{T}(ones(num_unknown) / num_unknown))
    else
        # number of 3-dimensional power cones needed is num_unknown - 1, number of new variables is num_unknown - 2
        # first num_unknown columns overlap with G_norm, column for the epigraph variable of the spectral cone added later
        G_geo = zeros(3 * (num_unknown - 1), 2 * num_unknown - 2)
        # first cone is a special case since two of the original variables participate in it
        G_geo[3, 1] = -1
        G_geo[2, 2] = -1
        G_geo[1, num_unknown + 1] = -1
        push!(cones, CO.HypoGeomean{T}([0.5, 0.5]))
        push!(cone_idxs, (cone_offset + 1):(cone_offset + 3))
        offset = 4
        # loop over new vars
        for i in 1:(num_unknown - 3)
            G_geo[offset, num_unknown + i + 1] = -1
            G_geo[offset + 1, num_unknown + i] = -1
            G_geo[offset + 2, i + 2] = -1
            push!(cones, CO.HypoGeomean{T}([(i + 1) / (i + 2), 1 / (i + 2)]))
            push!(cone_idxs, (cone_offset + 3 * i + 1):(cone_offset + 3 * (i + 1)))
            offset += 3
        end

        # last row also special becuase hypograph variable is fixed
        G_geo[offset + 2, num_unknown] = -1
        G_geo[offset + 1, 2 * num_unknown - 2] = -1
        push!(cones, CO.HypoGeomean{T}([(num_unknown - 1) / num_unknown, 1 / num_unknown]))
        push!(cone_idxs, (cone_offset + 3 * num_unknown - 5):(cone_offset + 3 * num_unknown - 3))
        h = vcat(h_norm, zeros(3 * (num_unknown - 2)), [1, 0, 0])

        # G_norm needs to be post-padded with columns for 3dim cone vars
        G_norm = [G_norm zeros(m * n + 1, num_unknown - 2)]
        # G_geo needs to be pre-padded with the epigraph variable for the spectral norm cone
        G_geo = [zeros(3 * (num_unknown - 1)) G_geo]
        c = vcat(c, zeros(num_unknown - 2))
        A = zeros(0, size(G_geo, 2))
    end
    G = vcat(G_norm, G_geo)

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

matrixcompletion_ex(use_geomean::Bool) = matrixcompletion(
    3,
    3,
    num_known = 5,
    known_rows = [1, 3, 2, 3, 1],
    known_cols = [1, 1, 2, 2, 3],
    known_vals = [1.0, 3.2, 0.8, 5.9, 1.9];
    use_geomean = true,
    use_epinorm = true,
    )
matrixcompletion1(; T::THR = Float64) = matrixcompletion_ex()
matrixcompletion2(; T::THR = Float64) = matrixcompletion_ex(use_geomean = false, use_epinorm = false)
matrixcompletion3(; T::THR = Float64) = matrixcompletion(5, 6)
matrixcompletion4(; T::THR = Float64) = matrixcompletion(5, 6, use_geomean = false)
matrixcompletion5(; T::THR = Float64) = matrixcompletion(5, 6, use_epinorm = false)
matrixcompletion6(; T::THR = Float64) = matrixcompletion(6, 8)
matrixcompletion7(; T::THR = Float64) = matrixcompletion(6, 8, use_geomean = false)
matrixcompletion8(; T::THR = Float64) = matrixcompletion(6, 8, use_epinorm = false)
matrixcompletion9(; T::THR = Float64) = matrixcompletion(8, 12)
matrixcompletion10(; T::THR = Float64) = matrixcompletion(8, 12, use_geomean = false)
matrixcompletion11(; T::THR = Float64) = matrixcompletion(8, 12, use_epinorm = false)

function test_matrixcompletion(instance::Function; T::THR = Float64, rseed::Int = 1, options)
    Random.seed!(rseed)
    d = instance(T = T)
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

test_matrixcompletion_all(; T::THR = Float64, options...) = test_matrixcompletion.([
    matrixcompletion1,
    matrixcompletion2,
    matrixcompletion3,
    matrixcompletion4,
    matrixcompletion5,
    matrixcompletion6,
    matrixcompletion7,
    matrixcompletion8,
    matrixcompletion9,
    matrixcompletion10,
    matrixcompletion11,
    ], T = T, options = options)

test_matrixcompletion(; T::THR = Float64, options...) = test_matrixcompletion.([
    matrixcompletion3,
    matrixcompletion4,
    matrixcompletion5,
    ], T = T, options = options)
