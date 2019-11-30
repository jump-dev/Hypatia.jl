#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

regularized matrix regression problems

TODO
- describe, references
=#

using LinearAlgebra
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
    # TODO row and column group lasso
    # lam_glr::T = zero(T),
    # lam_glc::T = zero(T),
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

    # loss function is minimize 1/2 squared frobenius norm of residuals, which we formulate with epipersquare
    # 1/2 * ||Y - X * A||_fro^2
    # = 1/2 * sum(abs2, R * A) - dot(X' * Y, A) + 1/2 * sum(abs2, Y)
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

    # put -dot(X' * Y, A) term in objective, ignore constant 1/2 * sum(abs2, Y)
    model_c = vcat(inv(T(2)), vec(-X' * Y))

    # list of optional regularizers
    # TODO think spectral has wrong ordering because data_p >= data_m
    reg_cone_dim = 1 + data_pm
    lams = [
        (lam_fro, CO.EpiNormEucl{T}(reg_cone_dim)), # frobenius norm (vector L2 norm)
        (lam_las, CO.EpiNormInf{T, T}(reg_cone_dim, true)), # vector lasso / L1 norm (dual to Linf norm)
        (lam_nuc, CO.EpiNormSpectral{T}(data_m, data_p, true)), # nuclear norm (dual to spectral norm)
        ]

    for (lam, cone) in lams
        if iszero(lam)
            continue
        end
        # obj term: lam * z
        # constraint: (z, data_A) in cone
        push!(model_c, lam)
        if cone isa CO.EpiNormSpectral{T}
            # permute identity because need transpose of data_A since data_p >= data_m
            iden_mat = zeros(T, data_pm, data_pm)
            # row_idx = 1
            # col_idx = 1
            for j in 1:data_m, k in 1:data_p
                iden_mat[j + (k - 1) * data_m, k + (j - 1) * data_p] = 1
            end
            @show iden_mat
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

    return (c = model_c, A = zeros(T, 0, model_n), b = zeros(T, 0), G = model_G, h = model_h, cones = cones)
end

# TODO remove toy problem
(n, m, p) = (5, 3, 4)
Y = rand(n, m)
X = rand(n, p)
A_try = X \ Y
# @show 1/2 * sum(abs2, Y - X * A_try) - 1/2 * sum(abs2, Y)
# @show sum(abs2, X * A_try)


matrixregression1(T::Type{<:Real}) = matrixregression(Y, X, lam_nuc = 0.1, lam_fro = 0.1, lam_las = 0.1)

instances_matrixregression_all = [
    matrixregression1,
    # matrixregression2,
    # matrixregression3,
    # matrixregression4,
    # matrixregression5,
    # matrixregression6,
    # matrixregression7,
    # matrixregression8,
    # matrixregression9,
    # matrixregression10,
    # matrixregression11,
    # matrixregression12,
    ]
instances_matrixregression_few = [
    matrixregression1,
    # matrixregression2,
    # matrixregression3,
    # matrixregression4,
    ]

function test_matrixregression(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    # @test r.status == :Optimal
    @show r.primal_obj
    A_opt = reshape(r.x[2:(1 + p * m)], p, m)
    @show 1/2 * sum(abs2, X * A_opt) - dot(X' * Y, A_opt)
    @show 1/2 * sum(abs2, X * A_opt) - dot(X' * Y, A_opt) + 0.1 * sum(svd(A_opt).S) + 0.1 * norm(A_opt, 2) + 0.1 * norm(A_opt, 1)
    @show A_opt
    return
end

test_matrixregression(matrixregression1)
