#=
regularized matrix regression problems

min 1/(2n) * ||Y - X * A||_fro^2 + lam_fro * ||A||_fro + lam_nuc * ||A||_nuc +
    lam_lass * ||A||_las + lam_glr * ||A||_glr + lamb_glc * ||A||_glc
- X is n x p
- Y is n x m
- A (variable) is p x m
- ||.||_fro is the Frobenius norm
- ||.||_nuc is the nuclear norm
- ||.||_las is the L1 norm
- ||.||_glr is the row group lasso penalty (L1 norm of row groups)
- ||.||_glc  is the column group lasso penalty (L1 norm of column groups)

TODO
- generalize for sparse Y,X but make sure qr factorization does not permute
=#

using SparseArrays

struct MatrixRegressionNative{T <: Real} <: ExampleInstanceNative{T}
    is_complex::Bool
    Y::Matrix
    X::Matrix
    lam_fro::Real # penalty on Frobenius norm
    lam_nuc::Real # penalty on nuclear norm
    lam_las::Real # penalty on l1 norm
    lam_glr::Real # penalty on penalty on row group l1 norm
    lam_glc::Real # penalty on penalty on column group l1 norm
end

function MatrixRegressionNative{T}(
    is_complex::Bool,
    n::Int,
    m::Int,
    p::Int,
    args...;
    A_max_rank::Int = div(m, 2) + 1,
    A_sparsity::Real = max(0.2, inv(sqrt(m * p))),
    Y_noise::Real = 0.01,
    ) where {T <: Real}
    @assert p >= m
    @assert 1 <= A_max_rank <= m
    @assert 0 < A_sparsity <= 1
    R = (is_complex ? Complex{T} : T)
    A_left = sprandn(R, p, A_max_rank, A_sparsity)
    A_right = sprandn(R, A_max_rank, m, A_sparsity)
    A = 10 * A_left * A_right
    X = randn(R, n, p)
    Y = Matrix{R}(X * A + Y_noise * randn(R, n, m))
    return MatrixRegressionNative{T}(is_complex, Y, X, args...)
end

function build(inst::MatrixRegressionNative{T}) where {T <: Real}
    (Y, X) = (inst.Y, inst.X)
    is_complex = inst.is_complex
    R = eltype(Y)
    @assert min(inst.lam_fro, inst.lam_nuc, inst.lam_las,
        inst.lam_glr, inst.lam_glc) >= 0
    (data_n, data_m) = size(Y)
    data_p = size(X, 2)
    @assert size(X, 1) == data_n
    @assert data_p >= data_m

    R_dim = (is_complex ? 2 : 1)
    data_pm = data_p * data_m
    data_nm = data_n * data_m
    # n is number of samples, m is number of responses, p is number of predictors
    # data_A (p * m) is matrix variable of coefficients, data_Y (n * m) is
    # response matrix, data_X (n * p) is design matrix
    model_n = 1 + R_dim * data_pm
    model_p = 0

    # loss function is 1/(2n) * ||Y - X * A||_fro^2
    # ||Y - X * A||^2 = tr((Y - X * A)' * (Y - X * A))
    # = tr((Y' - A' * X') * (Y - X * A))
    # = tr(A' * X' * X * A + Y' * Y - A' * X' * Y - Y' * X * A)
    # = tr(A' * X' * X * A) - 2 * tr(Y' * X * A) + tr(Y' * Y)
    # = sum(abs2, X * A) - 2 * dot(X' * Y, A) + sum(abs2, Y)
    # use EpiNormEucl for the sum(abs2, X * A) part
    # z >= ||Z||^2 is equivalent to (z, 1/2, vec(Z)) in EpiPerSquare is
    # equivalent to ((z + 1/2)/sqrt(2), (z - 1/2)/sqrt(2), vec(Z)) in EpiNormEucl
    if data_n > data_p
        # more samples than predictors: overdetermined
        # sum(abs2, X * A) = sum(abs2, cholesky(X' * X).U * A) =
        # sum(abs2, qr(X).R * A)
        # so can get a lower dimensional sum of squared terms
        Qhalf = qr(X).R
        # (Qhalf * A)_k,j = sum_k2 (Qhalf_k,k2 * A_k2,j)
        model_q = 2 + R_dim * data_pm
    else
        # fewer (or equal) samples than predictors
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
                    model_G[r_idx + 1, c_idx] = imag(Qlk2)
                    model_G[r_idx + 1, c_idx + 1] = real(Qlk2)
                    c_idx += 2
                end
                r_idx += 2
            else
                @. @views model_G[r_idx, c_idx:(c_idx + data_p - 1)] = Qhalf[l, :]
                r_idx += 1
                c_idx += data_p
            end
        end
    end

    cones = Cones.Cone{T}[Cones.EpiNormEucl{T}(model_q)]

    # put 1/(2n) * (epi(sum(abs2, X * A)) - 2 * dot(X' * Y, A)) in objective
    # ignore constant term 1/(2n) * sum(abs2, Y)
    model_c = zeros(T, model_n)
    mXpY = -X' * Y
    if is_complex
        @views Cones.vec_copyto!(model_c[2:end], vec(mXpY))
    else
        model_c[2:end] .= vec(mXpY)
    end
    model_c[1] = inv(T(2))
    model_c /= data_n

    # list of optional regularizers (group lasso handled separately below)
    reg_cone_dim = 1 + R_dim * data_pm
    lams = [
        (inst.lam_fro, Cones.EpiNormEucl{T}(reg_cone_dim)),
        (inst.lam_las, Cones.EpiNormInf{T, R}(reg_cone_dim, use_dual = true)),
        (inst.lam_nuc, Cones.EpiNormSpectral{T, R}(
            data_m, data_p, use_dual = true)),
        ]

    for (lam, cone) in lams
        if iszero(lam)
            continue # ignore regularizer because corresponding is zero
        end
        # obj term: lam * z
        # constraint: (z, data_A) in cone
        push!(model_c, lam)

        if cone isa Cones.EpiNormSpectral{T, R}
            # permute identity: need transpose of data_A since data_p >= data_m
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
            zeros(T, R_dim * data_pm, 1)  iden_mat  zeros(T, R_dim * data_pm,
                model_n - R_dim * data_pm);
            ]
        append!(model_h, zeros(reg_cone_dim))

        model_n += 1
        model_q += reg_cone_dim

        push!(cones, cone)
    end

    # row group lasso regularizer (one lambda for all rows)
    if !iszero(inst.lam_glr)
        append!(model_c, fill(inst.lam_glr, data_p))

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

        append!(cones, Cones.Cone{T}[Cones.EpiNormEucl{T}(
            1 + R_dim * data_m) for k in 1:data_p])
    end

    # column group lasso regularizer (one lambda for all columns)
    if !iszero(inst.lam_glc)
        append!(model_c, fill(inst.lam_glc, data_m))

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

        append!(cones, Cones.Cone{T}[Cones.EpiNormEucl{T}(1 + R_dim * data_p)
            for k in 1:data_m])
    end

    model = Models.Model{T}(model_c, zeros(T, 0, model_n), zeros(T, 0),
        model_G, model_h, cones)
    return model
end

function test_extra(
    inst::MatrixRegressionNative{T},
    solve_stats::NamedTuple,
    solution::NamedTuple,
    ) where T
    @test solve_stats.status == Solvers.Optimal
    (solve_stats.status == Solvers.Optimal) || return

    # check objective value is correct
    (Y, X) = (inst.Y, inst.X)
    A_opt = zeros(eltype(Y), size(X, 2), size(Y, 2))
    A_len = length(A_opt) * (inst.is_complex ? 2 : 1)
    @views Cones.vec_copyto!(A_opt, solution.x[1 .+ (1:A_len)])
    loss = (sum(abs2, X * A_opt) / 2 - real(dot(X' * Y, A_opt))) / size(Y, 1)
    obj_result = loss +
        inst.lam_fro * norm(vec(A_opt), 2) +
        inst.lam_nuc * sum(svd(A_opt).S) +
        inst.lam_las * norm(vec(A_opt), 1) +
        inst.lam_glr * sum(norm, eachrow(A_opt)) +
        inst.lam_glc * sum(norm, eachcol(A_opt))
    tol = eps(T)^0.25
    @test solve_stats.primal_obj ≈ obj_result atol=tol rtol=tol
    return
end
