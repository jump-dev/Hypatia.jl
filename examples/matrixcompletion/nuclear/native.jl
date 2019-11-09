#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

complete a matrix while minimizing the spectral norm

e.g.
Minimization of a Particular Singular Value by Alborz Alavian and Michael Rotkowitz
The Power of Convex Relaxation Emmanuel J. Candès and  Terence Tao
http://www.mit.edu/~parrilo/pubs/talkfiles/ISMP2009.pdf

possible extended formulations to (u, W) ∈ EpiNormSpectral(true):

lectures on convex programming (could replace m with any number of largest singular values)
u - ms - tr(Z) ≥ 0
Z ⪰ 0
Z - X + sI_m ⪰ 0
s ≥ 0
assumes X symmetric (generalization is probably replace Z with symmetrization of Z)

http://www.mit.edu/~parrilo/pubs/talkfiles/ISMP2009.pdf
min 1/2(tr(W1) + tr(W2))
[W1  X; X'  W2] ⪰ 0
=#

using LinearAlgebra
import Random
using Test
import Hypatia
const CO = Hypatia.Cones

function matrixcompletion(
    T::Type{<:Real},
    m::Int,
    n::Int;
    use_power::Bool = true,
    use_nuclearnorm::Bool = true,
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
    h_norm_x = zeros(T, mn)
    for (k, (i, j)) in enumerate(zip(known_rows, known_cols))
        known_idx = mat_to_vec_idx(i, j)
        # if not using the epinorminf cone, indices relate to X'
        h_norm_x[known_idx] = known_vals[k]
        is_known[known_idx] = true
    end

    num_known = sum(is_known) # if randomly generated, some indices may repeat
    num_unknown = mn - num_known
    b = T[]

    # dual epinormspectral cone- get vec(X) in G and h
    if use_nuclearnorm
        c = vcat(one(T), zeros(T, num_unknown))

        h_norm_x = vcat(zero(T), h_norm_x)
        h_norm = h_norm_x

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
        G = [
            -one(T)    zeros(T, 1, num_unknown);
            zeros(T, mn)    G_norm;
            ]
        h = h_norm
        A = zeros(T, 0, num_unknown + 1)

        cones = CO.Cone{T}[CO.EpiNormSpectral{T}(m, n, true)]
    else
        # X, W_1, W_2
        num_W1_vars = div(m * (m + 1), 2)
        num_W2_vars = div(n * (n + 1), 2)
        # TODO change to c_W1 = CO.mat_U_to_vec_scaled!(zeros(T, num_W1_vars), Diagonal(one(T) * I, m), sqrt(T(2)))
        c_W1 = CO.mat_U_to_vec_scaled!(zeros(T, num_W1_vars), Matrix{T}(I, m, m))
        c_W2 = CO.mat_U_to_vec_scaled!(zeros(T, num_W2_vars), Matrix{T}(I, n, n))

        c = vcat(zeros(T, num_unknown), c_W1, c_W2) / 2
        num_vars = num_W1_vars + num_W2_vars + num_unknown

        A = zeros(T, 0, num_vars)
        # unknown entries in X' unlike above
        num_rows = num_W1_vars + num_W2_vars + mn
        G_norm = zeros(T, num_rows, num_unknown + num_W1_vars + num_W2_vars)
        h = zeros(T, num_rows)
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
                    h[offset + idx] = h_norm_x[X_var_idx]
                end
            end
            # second block W_2
            for j in 1:i
                idx += 1
                W2_var_idx += 1
                G_norm[offset + idx, num_unknown + num_W1_vars + W2_var_idx] = -1
            end
        end
        G = G_norm
        cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(num_rows)]

    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

matrixcompletion1(T::Type{<:Real}) = matrixcompletion(T, 5, 6)
matrixcompletion2(T::Type{<:Real}) = matrixcompletion(T, 5, 6, use_nuclearnorm = false)
matrixcompletion3(T::Type{<:Real}) = matrixcompletion(T, 6, 8)
matrixcompletion4(T::Type{<:Real}) = matrixcompletion(T, 6, 8, use_nuclearnorm = false)
matrixcompletion5(T::Type{<:Real}) = matrixcompletion(T, 8, 8)
matrixcompletion6(T::Type{<:Real}) = matrixcompletion(T, 8, 8, use_nuclearnorm = false)

instances_matrixcompletion_all = [
    matrixcompletion1,
    matrixcompletion2,
    matrixcompletion3,
    matrixcompletion4,
    matrixcompletion5,
    matrixcompletion6,
    ]
instances_matrixcompletion_few = [
    matrixcompletion1,
    matrixcompletion2,
    matrixcompletion3,
    matrixcompletion4,
    ]

function test_matrixcompletion(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return
end
