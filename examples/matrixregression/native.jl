#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

regularized matrix regression problems

TODO
- describe, references
- generalize for sparse Y,X but make sure qr factorization does not permute
- generalize for complex cone
=#

using LinearAlgebra
using SparseArrays
import Random
using Test
import Hypatia
import Hypatia.RealOrComplex
const CO = Hypatia.Cones

function matrixregression(
    Y::Matrix{R},
    X::Matrix{R};
    lam_fro::Real = zero(T),
    lam_nuc::Real = zero(T),
    lam_las::Real = zero(T),
    lam_glr::Real = zero(T),
    lam_glc::Real = zero(T),
    ) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert lam_fro >= 0
    @assert lam_nuc >= 0
    @assert lam_las >= 0
    (data_n, data_m) = size(Y)
    data_p = size(X, 2)
    @assert size(X, 1) == data_n
    @assert data_p >= data_m

    is_complex = (R <: Complex{T})
    R_dim = (is_complex ? 2 : 1)
    data_pm = data_p * data_m
    data_nm = data_n * data_m
    # n is number of samples, m is number of responses, p is number of predictors
    # data_A (p * m) is matrix variable of coefficients, data_Y (n * m) is response matrix, data_X (n * p) is design matrix
    model_n = 1 + R_dim * data_pm
    model_p = 0

    # loss function is 1/(2n) * ||Y - X * A||_fro^2, which we formulate with epipersquare
    # ||Y - X * A||^2 = tr((Y - X * A)' * (Y - X * A)) = tr((Y' - A' * X') * (Y - X * A))
    # = tr(A' * X' * X * A + Y' * Y - A' * X' * Y - Y' * X * A)
    # = tr(A' * X' * X * A) - 2 * tr(Y' * X * A) + tr(Y' * Y)
    # = sum(abs2, X * A) - 2 * dot(X' * Y, A) + sum(abs2, Y)
    # use EpiNormEucl for the sum(abs2, X * A) part
    # z >= ||Z||^2 is equivalent to (z, 1/2, vec(Z)) in EpiPerSquare is equivalent to ((z + 1/2)/sqrt(2), (z - 1/2)/sqrt(2), vec(Z)) in EpiNormEucl
    if data_n > data_p
        # more samples than predictors: overdetermined
        # sum(abs2, X * A) = sum(abs2, cholesky(X' * X).U * A) = sum(abs2, qr(X).R * A)
        # so can get a lower dimensional sum of squared terms
        Qhalf = qr(X).R
        # (Qhalf * A)_k,j = sum_k2 (Qhalf_k,k2 * A_k2,j)
        model_q = 2 + R_dim * data_pm
    else
        # fewer (or equal) samples than predictors: underdetermined (or exactly determined)
        # X * A is already low dimensional
        Qhalf = X
        # (X * A)_i,j = sum_k2 (X_i,k2 * A_k2,j)
        model_q = 2 + R_dim * data_nm
    end

    model_h = zeros(T, model_q)
    rtT2 = sqrt(T(2))
    model_h[1] = rtT2 / 4
    model_h[2] = -model_h[1]

    model_G = zeros(T, model_q, model_n)
    model_G[1:2, 1] .= -inv(rtT2)
    r_idx = 3
    for l in 1:min(data_p, data_n)
        c_idx = 2
        for j in 1:data_m
            if is_complex
                for k2 in 1:data_p
                    Qlk2 = Qhalf[l, k2]
                    model_G[r_idx, c_idx] = real(Qlk2)
                    model_G[r_idx, c_idx + 1] = -imag(Qlk2)
                    r_idx += 1
                    model_G[r_idx, c_idx] = imag(Qlk2)
                    model_G[r_idx, c_idx + 1] = real(Qlk2)
                    r_idx += 1
                    c_idx += 2
                end
            else
                @. @views model_G[r_idx, c_idx:(c_idx + data_p - 1)] = Qhalf[l, :]
                r_idx += 1
                c_idx += data_p
            end
        end
    end

    cones = CO.Cone{T}[CO.EpiNormEucl{T}(model_q)]

    # put 1/(2n) * (epi(sum(abs2, X * A)) - 2 * dot(X' * Y, A)) in objective
    # ignore constant term 1/(2n) * sum(abs2, Y)
    model_c = zeros(T, model_n)
    mXpY = -X' * Y
    if is_complex
        @views cvec_to_rvec!(model_c[2:end], vec(mXpY))
    else
        model_c[2:end] .= vec(mXpY)
    end
    model_c[1] = inv(T(2))
    model_c /= data_n

    # list of optional regularizers (group lasso handled separately below)
    reg_cone_dim = 1 + R_dim * data_pm
    lams = [
        (lam_fro, CO.EpiNormEucl{T}(reg_cone_dim)), # frobenius norm (vector L2 norm)
        (lam_las, CO.EpiNormInf{T, R}(reg_cone_dim, true)), # vector lasso / L1 norm (dual to Linf norm)
        (lam_nuc, CO.EpiNormSpectral{T, R}(data_m, data_p, true)), # nuclear norm (dual to spectral norm)
        ]

    for (lam, cone) in lams
        if iszero(lam)
            continue # ignore regularizer because corresponding is zero
        end
        # obj term: lam * z
        # constraint: (z, data_A) in cone
        push!(model_c, lam)

        if cone isa CO.EpiNormSpectral{T, R}
            # permute identity because need transpose of data_A since data_p >= data_m
            iden_mat = zeros(T, R_dim * data_pm, R_dim * data_pm)
            for j in 1:data_m, k in 1:data_p
                if is_complex
                    r_idx = (2 * j - 1) + 2 * (k - 1) * data_m
                    c_idx = (2 * k - 1) + 2 * (j - 1) * data_p
                    iden_mat[r_idx, c_idx] = 1
                    iden_mat[r_idx + 1, c_idx + 1] = 1
                else
                    r_idx = j + (k - 1) * data_m
                    c_idx = k + (j - 1) * data_p
                    iden_mat[r_idx, c_idx] = 1
                end
            end
        else
            iden_mat = Matrix(-one(T) * I, R_dim * data_pm, R_dim * data_pm)
        end
        model_G = T[
            model_G  zeros(T, model_q, 1);
            zeros(T, 1, model_n)  -one(T);
            zeros(T, R_dim * data_pm, 1)  iden_mat  zeros(T, R_dim * data_pm, model_n - R_dim * data_pm);
            ]
        append!(model_h, zeros(reg_cone_dim))

        model_n += 1
        model_q += reg_cone_dim

        push!(cones, cone)
    end

    # row group lasso regularizer (one lambda for all rows)
    if !iszero(lam_glr)
        append!(model_c, fill(lam_glr, data_p))

        q_glr = data_p * (1 + R_dim * data_m)
        G_glr = zeros(T, q_glr, model_n + data_p)
        r_idx = 1
        for k in 1:data_p
            G_glr[r_idx, model_n + k] = -1
            r_idx += 1
            for j in 1:data_m
                if is_complex
                    c_idx = 1 + (2 * k - 1) + 2 * (j - 1) * data_p
                    G_glr[r_idx, c_idx] = -1
                    r_idx += 1
                    G_glr[r_idx, c_idx + 1] = -1
                    r_idx += 1
                else
                    c_idx = 1 + k + (j - 1) * data_p
                    G_glr[r_idx, c_idx] = -1
                    r_idx += 1
                end
            end
        end

        append!(model_h, zeros(q_glr))
        model_G = T[
            model_G  zeros(model_q, data_p);
            G_glr;
            ]

        model_n += data_p
        model_q += q_glr

        append!(cones, CO.Cone{T}[CO.EpiNormEucl{T}(1 + R_dim * data_m) for k in 1:data_p])
    end

    # column group lasso regularizer (one lambda for all columns)
    if !iszero(lam_glc)
        append!(model_c, fill(lam_glc, data_m))

        q_glc = data_m * (1 + R_dim * data_p)
        G_glc = zeros(T, q_glc, model_n + data_m)
        r_idx = 1
        for j in 1:data_m
            G_glc[r_idx, model_n + j] = -1
            r_idx += 1
            for k in 1:data_p
                if is_complex
                    c_idx = 1 + (2 * k - 1) + 2 * (j - 1) * data_p
                    G_glc[r_idx, c_idx] = -1
                    r_idx += 1
                    G_glc[r_idx, c_idx + 1] = -1
                    r_idx += 1
                else
                    c_idx = 1 + k + (j - 1) * data_p
                    G_glc[r_idx, c_idx] = -1
                    r_idx += 1
                end
            end
        end

        append!(model_h, zeros(q_glc))
        model_G = T[
            model_G  zeros(model_q, data_m);
            G_glc;
            ]

        model_n += data_m
        model_q += q_glc

        append!(cones, CO.Cone{T}[CO.EpiNormEucl{T}(1 + R_dim * data_p) for k in 1:data_m])
    end

    return (
        c = model_c, A = zeros(T, 0, model_n), b = zeros(T, 0), G = model_G, h = model_h, cones = cones,
        n = data_n, m = data_m, p = data_p, Y = Y, X = X,
        lam_fro = lam_fro, lam_nuc = lam_nuc, lam_las = lam_las, lam_glr = lam_glr, lam_glc = lam_glc
        )
end

function matrixregression(
    R::Type{RealOrComplex{T}},
    n::Int,
    m::Int,
    p::Int;
    A_max_rank::Int = div(m, 2) + 1,
    A_sparsity::Real = max(0.2, inv(sqrt(m * p))),
    Y_noise::Real = 0.01,
    model_kwargs...
    ) where {T <: Real}
    @assert p >= m
    @assert 1 <= A_max_rank <= m
    @assert 0 < A_sparsity <= 1

    A_left = sprandn(R, p, A_max_rank, A_sparsity)
    A_right = sprandn(R, A_max_rank, m, A_sparsity)
    A = 10 * A_left * A_right
    X = randn(R, n, p)
    Y = X * A + Y_noise * randn(R, n, m)

    Y = Matrix{R}(Y)
    X = Matrix{R}(X)
    A = Matrix{T}(A)
    @show A
    @show X
    @show Y

    return matrixregression(Y, X; model_kwargs...)
end

matrixregression1(R::Type{<:RealOrComplex}) = matrixregression(R, 5, 3, 4)
matrixregression2(R::Type{<:RealOrComplex}) = matrixregression(R, 5, 3, 4, lam_fro = 0.1, lam_nuc = 0.1, lam_las = 0.1, lam_glc = 0.2, lam_glr = 0.2)
matrixregression3(R::Type{<:RealOrComplex}) = matrixregression(R, 3, 4, 5)
matrixregression4(R::Type{<:RealOrComplex}) = matrixregression(R, 3, 4, 5, lam_fro = 0.1, lam_nuc = 0.1, lam_las = 0.1, lam_glc = 0.2, lam_glr = 0.2)
matrixregression5(R::Type{<:RealOrComplex}) = matrixregression(R, 100, 8, 12)
matrixregression6(R::Type{<:RealOrComplex}) = matrixregression(R, 100, 8, 12, lam_fro = 0.0, lam_nuc = 0.4, lam_las = 1.0, lam_glc = 0.1, lam_glr = 2.0)
matrixregression7(R::Type{<:RealOrComplex}) = matrixregression(R, 100, 8, 12, lam_fro = 0.0, lam_nuc = 0.0, lam_las = 0.0, lam_glc = 0.2, lam_glr = 1.5)
matrixregression8(R::Type{<:RealOrComplex}) = matrixregression(R, 15, 10, 20)
matrixregression9(R::Type{<:RealOrComplex}) = matrixregression(R, 15, 10, 20, lam_fro = 0.0, lam_nuc = 0.4, lam_las = 1.0, lam_glc = 0.1, lam_glr = 2.0)
matrixregression10(R::Type{<:RealOrComplex}) = matrixregression(R, 15, 10, 20, lam_fro = 0.0, lam_nuc = 0.0, lam_las = 0.0, lam_glc = 0.2, lam_glr = 1.5)

instances_matrixregression_all = [
    matrixregression1,
    matrixregression2,
    matrixregression3,
    matrixregression4,
    matrixregression5,
    matrixregression6,
    matrixregression7,
    matrixregression8,
    matrixregression9,
    matrixregression10,
    ]
instances_matrixregression_few = [
    matrixregression1,
    matrixregression2,
    matrixregression3,
    matrixregression4,
    ]

function test_matrixregression(instance::Function; R::Type{<:RealOrComplex{T}} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1) where {T <: Real}
    Random.seed!(rseed)
    d = instance(R)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    # check objective value is correct
    if R <: Complex
        A_opt_real = reshape(r.x[2:(1 + 2 * d.p * d.m)], 2 * d.p, d.m)
        A_opt = zeros(R, d.p, d.m)
        for k in 1:d.p
            @. @views A_opt[k, :] = A_opt_real[2k - 1, :] + A_opt_real[2k, :]
        end
    else
        A_opt = reshape(r.x[2:(1 + d.p * d.m)], d.p, d.m)
    end
    loss = (1/2 * sum(abs2, d.X * A_opt) - dot(d.X' * d.Y, A_opt)) / d.n
    # TODO don't calculate norms for which lambda is 0
    obj_try = loss + d.lam_fro * norm(vec(A_opt), 2) +
        d.lam_nuc * sum(svd(A_opt).S) + d.lam_las * norm(vec(A_opt), 1) +
        d.lam_glr * sum(norm, eachrow(A_opt)) + d.lam_glc * sum(norm, eachcol(A_opt))
    @test r.primal_obj ≈ obj_try atol = 1e-4 rtol = 1e-4
    # A_opt[abs.(A_opt) .< 1e-4] .= 0
    # @show A_opt
    return
end

# TODO delete
# @testset begin test_matrixregression.(instances_matrixregression_few) end
# @testset begin test_matrixregression.(instances_matrixregression_all) end
