#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

references minimizing the nuclear norm:
Minimization of a Particular Singular Value by Alborz Alavian and Michael Rotkowitz
The Power of Convex Relaxation Emmanuel J. Cand�s and  Terence Tao
http://www.mit.edu/~parrilo/pubs/talkfiles/ISMP2009.pdf
other:
https://www.cvxpy.org/examples/dgp/pf_matrix_completion.html

hypogeomean constraint inspired by:
https://www.cvxpy.org/examples/dgp/pf_matrix_completion.html

extended formulations to (u, W) in EpiNormSpectral(true) uses:
min 1/2(tr(W1) + tr(W2))
[W1 X; X' W2] ⪰ 0
from http://www.mit.edu/~parrilo/pubs/talkfiles/ISMP2009.pdf
=#

using LinearAlgebra
import Random
using Test
import Hypatia
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

function matrixcompletion(
    T::Type{<:Real},
    m::Int,
    n::Int;
    geomean_constr::Bool = true, # whether to add a constraint on the geomean of unknown values
    nuclearnorm_obj::Bool = true, # whether to use a nuclear norm as opposed to spectral norm in the objective
    use_hypogeomean::Bool = true, # natural/extended formulation for geomean constraint
    use_epinormspectral::Bool = true,  # natural/extended formulation for the objective
    )
    @assert m <= n
    mn = m * n

    num_known = round(Int, mn * 0.1)
    known_rows = rand(1:m, num_known)
    known_cols = rand(1:n, num_known)
    known_vals = rand(T, num_known) .- T(0.5)

    mat_to_vec_idx(i::Int, j::Int) = (j - 1) * m + i

    is_known = fill(false, mn)
    # h for the rows that X (the matrix and not epigraph variable) participates in
    h_norm_x = zeros(T, m * n)
    for (k, (i, j)) in enumerate(zip(known_rows, known_cols))
        known_idx = mat_to_vec_idx(i, j)
        # if not using the epinorminf cone, indices relate to X'
        h_norm_x[known_idx] = known_vals[k]
        is_known[known_idx] = true
    end

    num_known = sum(is_known) # if randomly generated, some indices may repeat
    num_unknown = m * n - num_known

    # epinormspectral cone or its dual- get vec(X) in G and h
    if use_epinormspectral
        c = vcat(one(T), zeros(T, num_unknown))
        G_norm = zeros(T, mn, num_unknown)
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
        G_norm = [
            -one(T)    zeros(T, 1, num_unknown);
            zeros(T, mn)    G_norm;
            ]
        h_norm_x = vcat(zero(T), h_norm_x)
        h_norm = h_norm_x

        cones = CO.Cone{T}[CO.EpiNormSpectral{T, T}(m, n, nuclearnorm_obj)]
    else
        # build an extended formulation for the norm used in the objective
        if nuclearnorm_obj
            # extended formulation for nuclear norm
            # X, W_1, W_2
            num_W1_vars = div(m * (m + 1), 2)
            num_W2_vars = div(n * (n + 1), 2)
            num_vars = num_W1_vars + num_W2_vars + num_unknown

            A = zeros(T, 0, num_vars)
            # unknown entries in X' unlike above
            num_rows = num_W1_vars + num_W2_vars + mn
            G_norm = zeros(T, num_rows, num_unknown + num_W1_vars + num_W2_vars)
            h_norm = zeros(T, num_rows)
            # first block W_1
            G_norm[1:num_W1_vars, (num_unknown + 1):(num_unknown + num_W1_vars)] = -Matrix{T}(I, num_W1_vars, num_W1_vars)

            offset = num_W1_vars
            # index to count rows in the bottom half of the large to-be-PSD matrix
            idx = 0
            # index only in X
            X_var_idx = 0
            W2_var_idx = 0
            # index of unknown vars (the x variables in the standard from), can increment it because we are moving row wise in X' (equivalent to columnwise in X)
            unknown_idx = 0
            # fill bottom `n` rows
            for i in 1:n
                # X'
                for j in 1:m
                    idx += 1
                    X_var_idx += 1
                    if !is_known[X_var_idx]
                        unknown_idx += 1
                        G_norm[offset + idx, unknown_idx] = -1
                    else
                        h_norm[offset + idx] = h_norm_x[X_var_idx]
                    end
                end
                # second block W_2
                for j in 1:i
                    idx += 1
                    W2_var_idx += 1
                    G_norm[offset + idx, num_unknown + num_W1_vars + W2_var_idx] = -1
                end
            end
            MU.vec_to_svec_cols!(G_norm, sqrt(T(2)))
            MU.vec_to_svec!(h_norm, sqrt(T(2)))
            cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(num_rows)]
            c_W1 = CO.smat_to_svec!(zeros(T, num_W1_vars), Diagonal(one(T) * I, m), sqrt(T(2)))
            c_W2 = CO.smat_to_svec!(zeros(T, num_W2_vars), Diagonal(one(T) * I, n), sqrt(T(2)))
            c = vcat(zeros(T, num_unknown), c_W1, c_W2) / 2
        else
            # extended formulation for spectral norm
            num_rows = div(m * (m + 1), 2) + m * n + div(n * (n + 1), 2)
            G_norm = zeros(T, num_rows, num_unknown + 1)
            h_norm = zeros(T, num_rows)
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
                        G_norm[offset + idx, 1 + unknown_idx] = -1
                        unknown_idx += 1
                    else
                        h_norm[offset + idx] = h_norm_x[var_idx]
                    end
                    idx += 1
                    var_idx += 1
                end
                # second block epigraph variable * I
                # skip `i` rows which will be filled with zeros
                idx += i
                G_norm[offset + idx - 1, 1] = -1
            end
            MU.vec_to_svec_cols!(G_norm, sqrt(T(2)))
            MU.vec_to_svec!(h_norm, sqrt(T(2)))
            cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(num_rows)]
            c = vcat(one(T), zeros(T, num_unknown))
        end
    end # objective natural true/false

    if geomean_constr
        if use_hypogeomean
            # hypogeomean for values to be filled
            G_geo = vcat(zeros(T, 1, num_unknown), Matrix{T}(-I, num_unknown, num_unknown))
            h = vcat(h_norm, one(T), zeros(T, num_unknown))

            # if using extended with spectral objective G_geo needs to be prepadded with an epigraph variable
            if nuclearnorm_obj
                if use_epinormspectral
                    prepad = zeros(T, num_unknown + 1, 1)
                    postpad = zeros(T, num_unknown + 1, 0)
                else
                    prepad = zeros(T, num_unknown + 1, 0)
                    postpad = zeros(T, num_unknown + 1, size(G_norm, 2) - num_unknown)
                end
            else
                prepad = zeros(T, num_unknown + 1, 1)
                postpad = zeros(T, num_unknown + 1, 0)
            end
            G = [
                G_norm;
                prepad  G_geo  postpad
                ]
            push!(cones, CO.HypoGeomean{T}(fill(inv(T(num_unknown)), num_unknown)))
        else
            # number of 3-dimensional power cones needed is num_unknown - 1, number of new variables is num_unknown - 2
            # first num_unknown columns overlap with G_norm, column for the epigraph variable of the spectral cone added later
            len_power = 3 * (num_unknown - 1)
            G_geo_unknown = zeros(T, len_power, num_unknown)
            G_geo_newvars = zeros(T, len_power, num_unknown - 2)
            # first cone is a special case since two of the original variables participate in it
            G_geo_unknown[1, 1] = -1
            G_geo_unknown[2, 2] = -1
            G_geo_newvars[3, 1] = -1
            push!(cones, CO.Power{T}(fill(inv(T(2)), 2), 1))
            offset = 4
            # loop over new vars
            for i in 1:(num_unknown - 3)
                G_geo_newvars[offset + 2, i + 1] = -1
                G_geo_newvars[offset + 1, i] = -1
                G_geo_unknown[offset, i + 2] = -1
                push!(cones, CO.Power{T}([inv(T(i + 2)), T(i + 1) / T(i + 2)], 1))
                offset += 3
            end

            # last row also special because hypograph variable is fixed
            G_geo_unknown[offset, num_unknown] = -1
            G_geo_newvars[offset + 1, num_unknown - 2] = -1
            push!(cones, CO.Power{T}([inv(T(num_unknown)), T(num_unknown - 1) / T(num_unknown)], 1))
            h = vcat(h_norm, zeros(T, 3 * (num_unknown - 2)), T[0, 0, 1])

            # if using extended with spectral objective G_geo needs to be prepadded with an epigraph variable
            if nuclearnorm_obj
                if use_epinormspectral
                    prepad = zeros(T, len_power, 1)
                    postpad = zeros(T, len_power, 0)
                else
                    prepad = zeros(T, len_power, 0)
                    postpad = zeros(T, len_power, size(G_norm, 2) - num_unknown)
                end
            else
                prepad = zeros(T, len_power, 1)
                postpad = zeros(T, len_power, 0)
            end
            G = [
                G_norm  zeros(T, size(G_norm, 1), num_unknown - 2);
                prepad  G_geo_unknown  postpad  G_geo_newvars
                ]

            c = vcat(c, zeros(T, num_unknown - 2))
        end # constraints natural true/false
    else
        G = G_norm
        h = h_norm
    end # add geomean constraint

    A = zeros(T, 0, size(G, 2))
    b = T[]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

matrixcompletion1(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = false)
matrixcompletion2(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = true)
matrixcompletion3(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = true, nuclearnorm_obj = false)
matrixcompletion4(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = true, nuclearnorm_obj = false, use_epinormspectral = false)
matrixcompletion5(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = true, nuclearnorm_obj = true)
matrixcompletion6(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = true, nuclearnorm_obj = true, use_epinormspectral = false)
matrixcompletion7(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = false, use_hypogeomean = false)
matrixcompletion8(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = true, nuclearnorm_obj = false, use_hypogeomean = false)
matrixcompletion9(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = true, nuclearnorm_obj = false, use_epinormspectral = false, use_hypogeomean = false)
matrixcompletion10(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = true, nuclearnorm_obj = true, use_hypogeomean = false)
matrixcompletion11(T::Type{<:Real}) = matrixcompletion(T, 2, 3, geomean_constr = true, nuclearnorm_obj = true, use_epinormspectral = false, use_hypogeomean = false)
matrixcompletion12(T::Type{<:Real}) = matrixcompletion(T, 6, 8, geomean_constr = true, nuclearnorm_obj = false, use_epinormspectral = false, use_hypogeomean = false)
matrixcompletion13(T::Type{<:Real}) = matrixcompletion(T, 6, 8, geomean_constr = true, nuclearnorm_obj = false, use_epinormspectral = true, use_hypogeomean = true)
matrixcompletion14(T::Type{<:Real}) = matrixcompletion(T, 6, 8, geomean_constr = true, nuclearnorm_obj = true, use_epinormspectral = false, use_hypogeomean = false)
matrixcompletion15(T::Type{<:Real}) = matrixcompletion(T, 6, 8, geomean_constr = true, nuclearnorm_obj = true, use_epinormspectral = true, use_hypogeomean = true)
matrixcompletion16(T::Type{<:Real}) = matrixcompletion(T, 12, 24, geomean_constr = true, nuclearnorm_obj = false, use_epinormspectral = false, use_hypogeomean = false)
matrixcompletion17(T::Type{<:Real}) = matrixcompletion(T, 12, 24, geomean_constr = true, nuclearnorm_obj = false, use_epinormspectral = true, use_hypogeomean = true)
matrixcompletion18(T::Type{<:Real}) = matrixcompletion(T, 12, 24, geomean_constr = true, nuclearnorm_obj = true, use_epinormspectral = false, use_hypogeomean = false)
matrixcompletion19(T::Type{<:Real}) = matrixcompletion(T, 12, 24, geomean_constr = true, nuclearnorm_obj = true, use_epinormspectral = true, use_hypogeomean = true)

instances_matrixcompletion_all = [
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
    matrixcompletion12,
    matrixcompletion13,
    matrixcompletion14,
    matrixcompletion15,
    matrixcompletion16,
    matrixcompletion17,
    matrixcompletion18,
    matrixcompletion19,
    ]
instances_matrixcompletion_few = [
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
    ]

function test_matrixcompletion(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return
end
