#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

regularized matrix regression problems

TODO
- describe, references
=#

using LinearAlgebra
using SparseArrays
import Random
using Test
import Hypatia
const CO = Hypatia.Cones

function matrixregression(
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T};
    lam_fro::T = zero(T),
    lam_nuc::T = zero(T),
    lam_las::T = zero(T),
    lam_glr::T = zero(T),
    lam_glc::T = zero(T),
    ) where {T <: Real}
    @assert lam_fro >= 0
    @assert lam_nuc >= 0
    @assert lam_las >= 0
    (data_n, data_m) = size(Y)
    data_p = size(X, 2)
    @assert size(X, 1) == data_n
    @assert data_n >= data_p >= data_m
    # n is number of samples, m is number of responses, p is number of predictors
    # data_A (p * m) is matrix variable of coefficients, data_Y (n * m) is response matrix, data_X (n * p) is design matrix

    # ||Y - X * A||^2 = tr((Y - X * A)' * (Y - X * A)) = tr((Y' - A' * X') * (Y - X * A))
    # = tr(A' * X' * X * A + Y' * Y - A' * X' * Y - Y' * X * A)
    # = tr(A' * X' * X * A) - 2 * tr(Y' * X * A) + tr(Y' * Y)
    # = sum(abs2, X * A) - 2 * dot(X' * Y, A) + sum(abs2, Y)
    # and sum(abs2, X * A) = sum(abs2, cholesky(X' * X).U * A) = sum(abs2, qr(X).R * A)
    # so can get a lower dimensional sum of squared terms
    R_factor = qr(X).R # TODO careful - assumes no pivoting TODO may be faster as cholesky(X' * X)

    # loss function is minimize 1/(2n) squared frobenius norm of residuals, which we formulate with epipersquare
    # 1/(2n) * ||Y - X * A||_fro^2
    # = 1/(2n) * sum(abs2, R * A) - 1/n * dot(X' * Y, A) + 1/(2n) * sum(abs2, Y)
    data_pm = data_p * data_m
    model_n = 1 + data_pm
    model_p = 0
    model_q = 1 + model_n

    # use EpiNormEucl for the sum(abs2, R * A) part
    # (R * A)_k,j = sum_i(R_k,k2 * A_k2,j)
    # z >= ||R * A||^2 is equivalent to (z, 1/2, vec(R * A)) in EpiPerSquare is equivalent to ((z + 1/2)/sqrt(2), (z - 1/2)/sqrt(2), vec(-R * A)) in EpiNormEucl
    model_h = zeros(T, model_q)
    rtT2 = sqrt(T(2))
    model_h[1] = rtT2 / 4
    model_h[2] = -model_h[1]
    model_G = zeros(T, model_q, model_n)
    model_G[1:2, 1] .= -inv(rtT2)
    r_idx = 3
    for k in 1:data_p
        c_idx = 2
        for j in 1:data_m
            @. @views model_G[r_idx, c_idx:(c_idx + data_p - 1)] = R_factor[k, :]
            r_idx += 1
            c_idx += data_p
        end
    end
    cones = CO.Cone{T}[CO.EpiNormEucl{T}(model_q)]

    # put -1/n * dot(X' * Y, A) term in objective, ignore constant 1/(2n) * sum(abs2, Y)
    model_c = vcat(inv(T(2)), vec(-X' * Y))
    model_c /= data_n

    # list of optional regularizers (group lasso handled separately below)
    reg_cone_dim = 1 + data_pm
    lams = [
        (lam_fro, CO.EpiNormEucl{T}(reg_cone_dim)), # frobenius norm (vector L2 norm)
        (lam_las, CO.EpiNormInf{T, T}(reg_cone_dim, true)), # vector lasso / L1 norm (dual to Linf norm)
        (lam_nuc, CO.EpiNormSpectral{T}(data_m, data_p, true)), # nuclear norm (dual to spectral norm)
        ]

    for (lam, cone) in lams
        if iszero(lam)
            continue # ignore regularizer because corresponding is zero
        end
        # obj term: lam * z
        # constraint: (z, data_A) in cone
        push!(model_c, lam)

        if cone isa CO.EpiNormSpectral{T}
            # permute identity because need transpose of data_A since data_p >= data_m
            iden_mat = zeros(T, data_pm, data_pm)
            for j in 1:data_m, k in 1:data_p
                iden_mat[j + (k - 1) * data_m, k + (j - 1) * data_p] = 1
            end
        else
            iden_mat = Matrix(-one(T) * I, data_pm, data_pm)
        end
        model_G = T[
            model_G  zeros(T, model_q, 1);
            zeros(T, 1, model_n)  -one(T);
            zeros(T, data_pm, 1)  iden_mat  zeros(T, data_pm, model_n - data_pm);
            ]
        append!(model_h, zeros(reg_cone_dim))

        model_n += 1
        model_q += reg_cone_dim

        push!(cones, cone)
    end

    # row group lasso regularizer (one lambda for all rows)
    if !iszero(lam_glr)
        append!(model_c, fill(lam_glr, data_p))

        q_glr = data_p * (1 + data_m)
        G_glr = zeros(T, q_glr, model_n + data_p)
        row_idx = 1
        for k in 1:data_p
            G_glr[row_idx, model_n + k] = -1
            row_idx += 1
            for j in 1:data_m
                G_glr[row_idx, 1 + k + (j - 1) * data_p] = -1
                row_idx += 1
            end
        end

        append!(model_h, zeros(q_glr))
        model_G = T[
            model_G  zeros(model_q, data_p);
            G_glr;
            ]

        model_n += data_p
        model_q += q_glr

        append!(cones, CO.Cone{T}[CO.EpiNormEucl{T}(data_m + 1) for k in 1:data_p])
    end

    # column group lasso regularizer (one lambda for all columns)
    if !iszero(lam_glc)
        append!(model_c, fill(lam_glc, data_m))

        q_glc = data_m * (1 + data_p)
        G_glc = zeros(T, q_glc, model_n + data_m)
        row_idx = 1
        for j in 1:data_m
            G_glc[row_idx, model_n + j] = -1
            row_idx += 1
            for k in 1:data_p
                G_glc[row_idx, 1 + k + (j - 1) * data_p] = -1
                row_idx += 1
            end
        end

        append!(model_h, zeros(q_glc))
        model_G = T[
            model_G  zeros(model_q, data_m);
            G_glc;
            ]

        model_n += data_m
        model_q += q_glc

        append!(cones, CO.Cone{T}[CO.EpiNormEucl{T}(data_p + 1) for k in 1:data_m])
    end

    return (
        c = model_c, A = zeros(T, 0, model_n), b = zeros(T, 0), G = model_G, h = model_h, cones = cones,
        n = data_n, m = data_m, p = data_p, Y = Y, X = X,
        lam_fro = lam_fro, lam_nuc = lam_nuc, lam_las = lam_las, lam_glr = lam_glr, lam_glc = lam_glc
        )
end

function matrixregression(
    T::Type{<:Real},
    n::Int,
    m::Int,
    p::Int;
    A_max_rank::Int = div(m, 2) + 1,
    A_sparsity::Real = max(0.2, inv(sqrt(m * p))),
    Y_noise::Real = 0.01,
    model_kwargs...
    )
    @assert n >= p >= m
    @assert 1 <= A_max_rank <= m
    @assert 0 < A_sparsity <= 1

    A_left = sprandn(T, p, A_max_rank, A_sparsity)
    A_right = sprandn(T, A_max_rank, m, A_sparsity)
    A = 10 * A_left * A_right
    X = randn(T, n, p)
    Y = X * A + Y_noise * randn(T, n, m)

    Y = Matrix{T}(Y)
    X = Matrix{T}(X)
    # A = Matrix{T}(A)
    # @show A
    # @show X
    # @show Y

    return matrixregression(Y, X; model_kwargs...)
end

matrixregression1(T::Type{<:Real}) = matrixregression(Float64, 5, 3, 4)
matrixregression2(T::Type{<:Real}) = matrixregression(Float64, 5, 3, 4, lam_fro = 0.1, lam_nuc = 0.1, lam_las = 0.1, lam_glc = 0.2, lam_glr = 0.2)
matrixregression3(T::Type{<:Real}) = matrixregression(Float64, 100, 8, 12)
matrixregression4(T::Type{<:Real}) = matrixregression(Float64, 100, 8, 12, lam_fro = 0.0, lam_nuc = 0.4, lam_las = 1.0, lam_glc = 0.1, lam_glr = 2.0)
matrixregression5(T::Type{<:Real}) = matrixregression(Float64, 100, 8, 12, lam_fro = 0.0, lam_nuc = 0.0, lam_las = 0.0, lam_glc = 0.2, lam_glr = 1.5)

instances_matrixregression_all = [
    matrixregression1,
    matrixregression2,
    matrixregression3,
    matrixregression4,
    matrixregression5,
    ]
instances_matrixregression_few = [
    matrixregression1,
    matrixregression2,
    ]

function test_matrixregression(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    # check objective value is correct
    A_opt = reshape(r.x[2:(1 + d.p * d.m)], d.p, d.m)
    loss = (1/2 * sum(abs2, d.X * A_opt) - dot(d.X' * d.Y, A_opt)) / d.n
    # TODO don't calculate norms for which lambda is 0
    obj_try = loss + d.lam_fro * norm(vec(A_opt), 2) + d.lam_nuc * sum(svd(A_opt).S) + d.lam_las * norm(vec(A_opt), 1) + d.lam_glr * sum(norm, eachrow(A_opt)) + d.lam_glc * sum(norm, eachcol(A_opt))
    @test r.primal_obj ≈ obj_try atol = 1e-4 rtol = 1e-4
    # A_opt[abs.(A_opt) .< 1e-4] .= 0
    # @show A_opt
    return
end
