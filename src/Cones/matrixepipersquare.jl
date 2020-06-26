#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

matrix epigraph of matrix square

(U, v, W) in (S_+^n, R_+, R^(n, m)) such that 2 * U * v - W * W' in S_+^n
=#

mutable struct MatrixEpiPerSquare{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    d1::Int
    d2::Int
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}
    rt2::T
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    U_idxs::UnitRange{Int}
    v_idx::Int
    W_idxs::UnitRange{Int}
    U::Hermitian{R, Matrix{R}}
    dual_U::Hermitian{R, Matrix{R}}
    W::Matrix{R}
    dual_W::Matrix{R}
    Z::Hermitian{R, Matrix{R}}
    fact_Z
    Zi::Hermitian{R, Matrix{R}}
    ZiW::Matrix{R}
    ZiUZi::Hermitian{R, Matrix{R}}
    tmpmm::Matrix{R}
    tmpnn::Matrix{R}

    function MatrixEpiPerSquare{T, R}(
        d1::Int,
        d2::Int;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert 1 <= d1 <= d2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.is_complex = (R <: Complex)
        cone.v_idx = (cone.is_complex ? d1 ^ 2 + 1 : svec_length(d1) + 1)
        cone.dim = cone.v_idx + (cone.is_complex ? 2 : 1) * d1 * d2
        cone.d1 = d1
        cone.d2 = d2
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        cone.U_idxs = 1:(cone.v_idx - 1)
        cone.W_idxs = (cone.v_idx + 1):cone.dim
        return cone
    end
end

# TODO only allocate the fields we use
function setup_data(cone::MatrixEpiPerSquare{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    d1 = cone.d1
    d2 = cone.d2
    cone.U = Hermitian(zeros(R, d1, d1), :U)
    cone.dual_U = Hermitian(zeros(R, d1, d1), :U)
    cone.W = zeros(R, d1, d2)
    cone.dual_W = zeros(R, d1, d2)
    cone.Z = Hermitian(zeros(R, d1, d1), :U)
    cone.ZiW = Matrix{R}(undef, d1, d2)
    cone.ZiUZi = Hermitian(zeros(R, d1, d1), :U)
    cone.tmpmm = Matrix{R}(undef, d2, d2)
    cone.tmpnn = Matrix{R}(undef, d1, d1)
    return
end

get_nu(cone::MatrixEpiPerSquare) = cone.d1 + 1

use_correction(cone::MatrixEpiPerSquare) = true

function set_initial_point(arr::AbstractVector, cone::MatrixEpiPerSquare{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    @inbounds for i in 1:cone.d1
        arr[k] = 1
        k += incr * i + 1
    end
    arr[cone.v_idx] = 1
    return arr
end

function update_feas(cone::MatrixEpiPerSquare)
    @assert !cone.feas_updated
    v = cone.point[cone.v_idx]

    if v > 0
        @views U = svec_to_smat!(cone.U.data, cone.point[cone.U_idxs], cone.rt2)
        @views W = vec_copy_to!(cone.W, cone.point[cone.W_idxs])
        copyto!(cone.Z.data, U)
        mul!(cone.Z.data, cone.W, cone.W', -1, 2 * v)
        cone.fact_Z = cholesky!(cone.Z, check = false)
        cone.is_feas = isposdef(cone.fact_Z)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_dual_feas(cone::MatrixEpiPerSquare)
    v = cone.dual_point[cone.v_idx]
    if v > 0
        @views U = svec_to_smat!(cone.dual_U.data, cone.dual_point[cone.U_idxs], cone.rt2)
        F = cholesky!(cone.dual_U, check = false)
        isposdef(F) || return false
        @views W = vec_copy_to!(cone.dual_W, cone.dual_point[cone.W_idxs])
        LW = ldiv!(F.L, W)
        trLW = sum(abs2, LW)
        if 2 * v >= trLW
            return true
        end
    end
    return false
end

function update_grad(cone::MatrixEpiPerSquare)
    @assert cone.is_feas
    U = cone.U
    W = cone.W
    dim = cone.dim
    v = cone.point[cone.v_idx]

    Zi = cone.Zi = Hermitian(inv(cone.fact_Z), :U)
    @views smat_to_svec!(cone.grad[cone.U_idxs], Zi, cone.rt2)
    @views cone.grad[cone.U_idxs] .*= -2v
    cone.grad[cone.v_idx] = -2 * dot(Zi, U) + (cone.d1 - 1) / v
    ldiv!(cone.ZiW, cone.fact_Z, W)
    @views vec_copy_to!(cone.grad[cone.W_idxs], cone.ZiW)
    @. @views cone.grad[cone.W_idxs] *= 2

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::MatrixEpiPerSquare)
    @assert cone.grad_updated
    d1 = cone.d1
    d2 = cone.d2
    dim = cone.dim
    U_idxs = cone.U_idxs
    v_idx = cone.v_idx
    W_idxs = cone.W_idxs
    U = cone.U
    W = cone.W
    v = cone.point[cone.v_idx]
    H = cone.hess.data
    tmpmm = cone.tmpmm
    ZiUZi = cone.ZiUZi
    tmpnn = cone.tmpnn
    Zi = cone.Zi
    ZiW = cone.ZiW
    idx_incr = (cone.is_complex ? 2 : 1)

    # H_W_W part
    mul!(tmpmm, W', ZiW)
    tmpmm += I # TODO inefficient

    # TODO parallelize loops
    r_idx = v_idx + 1
    for i in 1:d2, j in 1:d1
        c_idx = r_idx
        @inbounds for k in i:d2
            ZiWjk = ZiW[j, k]
            tmpmmik = tmpmm[i, k]
            lstart = (i == k ? j : 1)
            @inbounds for l in lstart:d1
                term1 = Zi[l, j] * tmpmmik
                term2 = ZiW[l, i] * ZiWjk
                hess_element(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
        r_idx += idx_incr
    end
    @views H[W_idxs, W_idxs] .*= 2

    # H_U_U part
    @views H_U_U = H[U_idxs, U_idxs]
    symm_kron(H_U_U, Zi, cone.rt2)
    @. @views H_U_U *= 4 * abs2(v)

    # H_v_v part
    ldiv!(ZiUZi.data, cone.fact_Z, U)
    rdiv!(ZiUZi.data, cone.fact_Z)
    @views H[v_idx, v_idx] = 4 * dot(ZiUZi, U) - (d1 - 1) / v / v

    # H_U_W part
    # TODO parallelize loops
    # TODO use dispatch for complex part and clean up
    row_idx = 1
    for i in 1:d1, j in 1:i # U lower tri idxs
        col_idx = v_idx + 1
        for l in 1:d2, k in 1:d1 # W idxs
            @inbounds if cone.is_complex
                term1 = Zi[k, i] * ZiW[j, l]
                term2 = Zi[k, j] * ZiW[i, l]
                if i != j
                    term1 *= cone.rt2
                    term2 *= cone.rt2
                end
                H[row_idx, col_idx] = real(term1) + real(term2)
                H[row_idx, col_idx + 1] = imag(term1) + imag(term2)
                if i != j
                    H[row_idx + 1, col_idx] = imag(term2) - imag(term1)
                    H[row_idx + 1, col_idx + 1] = real(term1) - real(term2)
                end
            else
                term = Zi[i, k] * ZiW[j, l] + Zi[k, j] * ZiW[i, l]
                if i != j
                    term *= cone.rt2
                end
                H[row_idx, col_idx] = term
            end
            col_idx += idx_incr
        end
        if i != j
            row_idx += idx_incr
        else
            row_idx += 1
        end
    end
    @. @views H[U_idxs, W_idxs] *= -2v

    # H_v_W part
    # NOTE overwrites ZiW
    # TODO better to do ZiU * ZiW?
    mul!(ZiW, ZiUZi, W, -4, false)
    @views vec_copy_to!(H[v_idx, W_idxs], ZiW)

    # H_U_v part
    copyto!(tmpnn, ZiUZi)
    axpby!(-2, Zi, 4v, tmpnn)
    @views smat_to_svec!(H[U_idxs, v_idx], tmpnn, cone.rt2)

    cone.hess_updated = true
    return cone.hess
end

# TODO reduce allocs
function correction2(cone::MatrixEpiPerSquare, primal_dir::AbstractVector)
    @assert cone.hess_updated
    d1 = cone.d1
    d2 = cone.d2
    dim = cone.dim

    U_idxs = cone.U_idxs
    v_idx = cone.v_idx
    W_idxs = cone.W_idxs
    U = cone.U
    W = cone.W
    v = cone.point[cone.v_idx]

    Zi = Hermitian(cone.Zi)
    tau = Zi * W # TODO don't do that, cache, rename
    Wtau = W' * tau # TODO don't do that, cache, rename
    ZiUZi = cone.ZiUZi
    ZiUZiUZi = Hermitian(ZiUZi * U * Zi)
    T = Float64
    v_idx = cone.v_idx

    third = zeros(dim, dim, dim)
    third_debug = zeros(dim, dim, dim)
    corr = cone.correction
    corr .= 0
    U_corr = view(corr, U_idxs)
    W_corr = view(corr, W_idxs)
    @views U_dir = Hermitian(svec_to_smat!(similar(U.data), primal_dir[U_idxs], cone.rt2))
    @views W_dir = vec_copy_to!(similar(W), primal_dir[W_idxs])
    v_dir = primal_dir[v_idx]

    ZiU = cone.fact_Z \ U # TODO cache or don't use
    ZiUZiW = ZiU * tau

    # uvv
    U_corr .+= 8 * abs2(v_dir) * smat_to_svec!(similar(U_corr), ZiUZi - 2 * v * ZiUZiUZi, cone.rt2)
    corr[v_idx] += 16 * v_dir * dot(ZiUZi - 2 * v * ZiUZiUZi, U_dir)

    # uvw term 1, kron using Zi and tau
    t1_uvw_W = Zi * W_dir * tau' + (Zi * W_dir * tau')'  # TODO could do reuse either of these two in corr[v_idx]
    t1_uvw_U = Zi * U_dir * tau * 2
    U_corr .+= -4 * v_dir * smat_to_svec!(similar(U_corr), t1_uvw_W, cone.rt2)
    # corr[v_idx] += -4 * dot(U_dir, t1_uvw_W)
    corr[v_idx] += -4 * real(dot(W_dir, t1_uvw_U))
    W_corr .+= -4 * v_dir * vec_copy_to!(similar(W_corr), t1_uvw_U)
    # uvw term 2, kron using ZiUZi and tau times v
    t2_uvw_W = ZiUZi * W_dir * tau' + (ZiUZi * W_dir * tau')'  # TODO could do reuse either of these two in corr[v_idx]
    t2_uvw_U = ZiUZi * U_dir * tau * 2
    U_corr .+= 8 * v * v_dir * smat_to_svec!(similar(U_corr), t2_uvw_W, cone.rt2)
    # corr[v_idx] += 8 * v * dot(U_dir, t2_uvw_W)
    corr[v_idx] += 8 * v * real(dot(W_dir, t2_uvw_U))
    W_corr .+= 8 * v * v_dir * vec_copy_to!(similar(W_corr), t2_uvw_U)
    # uvw term 3, kron using ZiUZiW and Zi
    t3_uvw_W = Zi * W_dir * ZiUZiW' + (Zi * W_dir * ZiUZiW')'  # TODO could do reuse either of these two in corr[v_idx]
    t3_uvw_U = Zi * U_dir * ZiUZiW * 2
    U_corr .+= 8 * v * v_dir * smat_to_svec!(similar(U_corr), t3_uvw_W, cone.rt2)
    # corr[v_idx] += 8 * v * dot(U_dir, t3_uvw_W)
    corr[v_idx] += 8 * v * real(dot(W_dir, t3_uvw_U))
    W_corr .+= 8 * v * v_dir * vec_copy_to!(similar(W_corr), t3_uvw_U)

    # uww term 1, kron using Zi, tau, tau
    t1_uww_W = Zi * W_dir * tau' * W_dir * tau' + tau * W_dir' * Zi * W_dir * tau' + (Zi * W_dir * tau' * W_dir * tau')'
    U_corr .+= -4 * v * smat_to_svec!(similar(U_corr), t1_uww_W, cone.rt2)
    t1_uww_U =
        Zi * U_dir * tau * W_dir' * tau +
        Zi * W_dir * tau' * U_dir * tau +
        tau * W_dir' * Zi * U_dir * tau
    W_corr .+= -8 * v * vec_copy_to!(similar(W_corr), t1_uww_U)
    # uww term 2, kron using Wtau, Zi, Zi and kron using I, Zi, Zi
    t2_uww_W = Zi * W_dir * Wtau * W_dir' * Zi + Zi * W_dir * W_dir' * Zi
    U_corr .+= -4 * v * smat_to_svec!(similar(U_corr), t2_uww_W, cone.rt2)
    t2_uww_U = Zi * U_dir * Zi * W_dir * Wtau + Zi * U_dir * Zi * W_dir
    W_corr .+= -8 * v * vec_copy_to!(similar(W_corr), t2_uww_U)

    # uuv term1 kron of Zi and Zi
    t1_uuv_U = Zi * U_dir * Zi
    U_corr .+= 16 * v * v_dir * smat_to_svec!(similar(U_corr), t1_uuv_U, cone.rt2)
    corr[v_idx] += 8 * v * real(dot(U_dir, t1_uuv_U))
    # uuv term2 kron of ZiUZi and Zi
    t2_uuv_U = ZiUZi * U_dir * Zi + Zi * U_dir * ZiUZi
    U_corr .+= -16 * abs2(v) * v_dir * smat_to_svec!(similar(U_corr), t2_uuv_U, cone.rt2)
    corr[v_idx] += -8 * abs2(v) * real(dot(U_dir, t2_uuv_U))

    # uuu
    U_corr .+= -16 * v ^ 3 * smat_to_svec!(similar(U_corr), Zi * U_dir * Zi * U_dir * Zi, cone.rt2)

    # uuw
    uuv_UW = Zi * U_dir * Zi * W_dir * tau' + (Zi * U_dir * Zi * W_dir * tau')' + Zi * W_dir * tau' * U_dir * Zi + (Zi * W_dir * tau' * U_dir * Zi)'
    U_corr .+= 8 * abs2(v) * smat_to_svec!(similar(U_corr), uuv_UW, cone.rt2)
    uuv_UU = Zi * U_dir * Zi * U_dir * tau
    W_corr .+= 16 * abs2(v) * vec_copy_to!(similar(W_corr), uuv_UU)

    # Tvvv
    third[v_idx, v_idx, v_idx] = -16 * real(tr(ZiU ^ 3)) + 2 * (d1 - 1) / v ^ 3

    vvv = -16 * real(tr(ZiU ^ 3)) + 2 * (d1 - 1) / v ^ 3

    corr[v_idx] += vvv * abs2(v_dir)

    # www
    # copied from spectral norm cone
    WtauI = W' * tau + I
    Wdirtau = W_dir' * tau
    ZiWdir = cone.fact_Z \ W_dir
    ZiWdirWtauI = ZiWdir * WtauI
    terms_twww = 4 * (
        tau * (Wdirtau * Wdirtau + W_dir' * ZiWdirWtauI) +
        ZiWdirWtauI * Wdirtau +
        ZiWdir * Wdirtau' * WtauI
        )
    W_corr .+= vec_copy_to!(similar(W_corr), terms_twww)

    # vvw
    vvw_W = ZiUZiUZi * W
    corr[v_idx] += 32 * v_dir * real(dot(vvw_W, W_dir))
    W_corr .+= 16 * abs2(v_dir) * vec_copy_to!(similar(W_corr), vvw_W)

    # vww
    # term 1 kron of ZiUZi and Wtau plus kron of ZiUZi and I
    t1_vvw_W = ZiUZi * W_dir * Wtau + ZiUZi * W_dir
    corr[v_idx] += -4 * real(dot(t1_vvw_W, W_dir))
    W_corr .+= -8 * v_dir * vec_copy_to!(similar(W_corr), t1_vvw_W)
    # term 2 kron of Zi and (W' * ZiU * tau)
    t2_vvw_W = Zi * W_dir * (W' * ZiU * tau)
    corr[v_idx] += -4 * real(dot(t2_vvw_W, W_dir))
    W_corr .+= -8 * v_dir * vec_copy_to!(similar(W_corr), t2_vvw_W)
    # term 3 kron of (ZiU * tau)' and tau
    t3_vvw_W = (ZiU * tau) * W_dir' * tau + tau * W_dir' * (ZiU * tau)
    corr[v_idx] += -4 * real(dot(t3_vvw_W, W_dir))
    W_corr .+= -8 * v_dir * vec_copy_to!(similar(W_corr), t3_vvw_W)

    corr ./= -2

    return corr
end



# # TODO for experimenting with jordan hessian / inverse-hessian products like S * vec * S
# function symmat(s::AbstractVector{T}, d1, d2) where {T <: Real}
#     @assert d1 <= d2
#     side = d1 + d2
#     v_idx = svec_length(d1) + 1
#     S = zeros(T, side, side)
#     @views svec_to_smat!(S[1:d1, 1:d1], s[1:(v_idx - 1)], sqrt(T(2)))
#     S[(d1 + 1):end, (d1 + 1):end] += 2 * s[v_idx] * I
#     @views vec_copy_to!(S[1:d1, (d1 + 1):end], s[(v_idx + 1):end])
#     S = Symmetric(S, :U)
#     return M
# end
# function symvec(S::AbstractMatrix{T}, d1, d2) where {T <: Real}
#     @assert d1 <= d2
#     v_idx = svec_length(d1) + 1
#     s = zeros(T, v_idx + 1 + d1 * d2)
#     @views smat_to_svec!(s[1:(v_idx - 1)], S[1:d1, 1:d1], sqrt(T(2)))
#     s[v_idx] = M[end, end] / 2
#     @views vec_copy_to!(s[(v_idx + 1):end], S[1:d1, (d1 + 1):end])
#     return v
# end
