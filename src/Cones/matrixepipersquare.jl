"""
$(TYPEDEF)

Matrix epigraph of perspective function of real or complex matrix outer product
for a matrix (stacked column-wise) of `nrows` rows and `ncols` columns with
`nrows ≤ ncols`.

    $(FUNCTIONNAME){T, R}(nrows::Int, ncols::Int, use_dual::Bool = false)
"""
mutable struct MatrixEpiPerSquare{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d1::Int
    d2::Int
    is_complex::Bool
    rt2::T

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    hess_aux_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    U_idxs::UnitRange{Int}
    v_idx::Int
    W_idxs::UnitRange{Int}
    U::Hermitian{R, Matrix{R}}
    W::Matrix{R}
    Z::Hermitian{R, Matrix{R}}
    fact_Z
    Zi::Hermitian{R, Matrix{R}}
    ZiW::Matrix{R}
    ZiUZi::Hermitian{R, Matrix{R}}
    WtZiW::Hermitian{R, Matrix{R}}
    tempd2d2::Matrix{R}
    tempd1d1::Matrix{R}
    tempd1d1b::Matrix{R}
    tempd1d1c::Matrix{R}
    tempd1d1d::Matrix{R}
    tempd1d2::Matrix{R}
    ZiUZiW::Matrix{R}
    Hvv::T

    function MatrixEpiPerSquare{T, R}(
        d1::Int,
        d2::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert 1 <= d1 <= d2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.is_complex = (R <: Complex)
        cone.v_idx = 1 + svec_length(R, d1)
        cone.dim = cone.v_idx + vec_length(R, d1 * d2)
        cone.d1 = d1
        cone.d2 = d2
        cone.rt2 = sqrt(T(2))
        cone.U_idxs = 1:(cone.v_idx - 1)
        cone.W_idxs = (cone.v_idx + 1):cone.dim
        return cone
    end
end

reset_data(cone::MatrixEpiPerSquare) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated =
    cone.hess_aux_updated = false)

function setup_extra_data!(
    cone::MatrixEpiPerSquare{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    (d1, d2) = (cone.d1, cone.d2)
    cone.U = Hermitian(zeros(R, d1, d1), :U)
    cone.W = zeros(R, d1, d2)
    cone.Z = Hermitian(zeros(R, d1, d1), :U)
    cone.Zi = Hermitian(zeros(R, d1, d1), :U)
    cone.ZiW = zeros(R, d1, d2)
    cone.ZiUZi = Hermitian(zeros(R, d1, d1), :U)
    cone.WtZiW = Hermitian(zeros(R, d2, d2), :U)
    cone.tempd2d2 = zeros(R, d2, d2)
    cone.tempd1d1 = zeros(R, d1, d1)
    cone.tempd1d1b = zeros(R, d1, d1)
    cone.tempd1d1c = zeros(R, d1, d1)
    cone.tempd1d1d = zeros(R, d1, d1)
    cone.tempd1d2 = zeros(R, d1, d2)
    cone.ZiUZiW = zeros(R, d1, d2)
    return cone
end

get_nu(cone::MatrixEpiPerSquare) = cone.d1 + 1

function set_initial_point!(
    arr::AbstractVector,
    cone::MatrixEpiPerSquare{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
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

function update_feas(cone::MatrixEpiPerSquare{T}) where T
    @assert !cone.feas_updated
    v = cone.point[cone.v_idx]

    if v > eps(T)
        @views U = svec_to_smat!(cone.U.data, cone.point[cone.U_idxs], cone.rt2)
        @views W = vec_copyto!(cone.W, cone.point[cone.W_idxs])
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

function is_dual_feas(cone::MatrixEpiPerSquare{T}) where T
    v = cone.dual_point[cone.v_idx]
    if v > eps(T)
        @views svec_to_smat!(cone.tempd1d1b, cone.dual_point[cone.U_idxs],
            cone.rt2)
        F = cholesky!(Hermitian(cone.tempd1d1b, :U), check = false)
        isposdef(F) || return false
        @views W = vec_copyto!(cone.tempd1d2, cone.dual_point[cone.W_idxs])
        LW = ldiv!(F.U', W)
        trLW = sum(abs2, LW)
        return (2 * v - trLW > eps(T))
    end
    return false
end

function update_grad(cone::MatrixEpiPerSquare)
    @assert cone.is_feas
    U = cone.U
    W = cone.W
    dim = cone.dim
    v = cone.point[cone.v_idx]

    Zi = cone.Zi
    inv_fact!(Zi.data, cone.fact_Z)
    @views smat_to_svec!(cone.grad[cone.U_idxs], Zi, cone.rt2)
    @views cone.grad[cone.U_idxs] .*= -2 * v
    cone.grad[cone.v_idx] = -2 * dot(Zi, U) + (cone.d1 - 1) / v
    ldiv!(cone.ZiW, cone.fact_Z, W)
    @views vec_copyto!(cone.grad[cone.W_idxs], cone.ZiW)
    @. @views cone.grad[cone.W_idxs] *= 2

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::MatrixEpiPerSquare)
    @assert cone.grad_updated
    ZiUZi = cone.ZiUZi

    ldiv!(ZiUZi.data, cone.fact_Z, cone.U)
    rdiv!(ZiUZi.data, cone.fact_Z)
    mul!(cone.WtZiW.data, cone.W', cone.ZiW) # TODO not used for hess prod
    v = cone.point[cone.v_idx]
    cone.Hvv = 4 * dot(ZiUZi, cone.U) - (cone.d1 - 1) / v / v
    mul!(cone.ZiUZiW, cone.ZiUZi, cone.W)

    cone.hess_aux_updated = true
    return cone.hess_aux_updated
end

function update_hess(cone::MatrixEpiPerSquare)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    d1 = cone.d1
    d2 = cone.d2
    U_idxs = cone.U_idxs
    v_idx = cone.v_idx
    W_idxs = cone.W_idxs
    v = cone.point[v_idx]
    tempd2d2 = cone.tempd2d2
    ZiUZi = cone.ZiUZi
    tempd1d1 = cone.tempd1d1
    Zi = cone.Zi
    ZiW = cone.ZiW
    idx_incr = (cone.is_complex ? 2 : 1)

    # H_W_W part
    copyto!(tempd2d2, I)
    tempd2d2 .+= cone.WtZiW

    # TODO parallelize loops
    r_idx = v_idx + 1
    for i in 1:d2, j in 1:d1
        c_idx = r_idx
        @inbounds for k in i:d2
            ZiWjk = ZiW[j, k]
            tempd2d2ik = tempd2d2[i, k]
            lstart = (i == k ? j : 1)
            @inbounds for l in lstart:d1
                term1 = Zi[l, j] * tempd2d2ik
                term2 = ZiW[l, i] * ZiWjk
                spectral_kron_element!(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
        r_idx += idx_incr
    end
    @views H[W_idxs, W_idxs] .*= 2

    # H_U_U part
    @views H_U_U = H[U_idxs, U_idxs]
    symm_kron!(H_U_U, Zi, cone.rt2)
    @. H_U_U *= 4 * abs2(v)

    # H_v_v part
    @views H[v_idx, v_idx] = cone.Hvv

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
        row_idx += (i == j ? 1 : idx_incr)
    end
    @. @views H[U_idxs, W_idxs] *= -2 * v

    # H_v_W part
    @views vec_copyto!(H[v_idx, W_idxs], cone.ZiUZiW)
    @. @views H[v_idx, W_idxs] *= -4

    # H_U_v part
    copyto!(tempd1d1, ZiUZi)
    axpby!(-2, Zi, 4 * v, tempd1d1)
    @views smat_to_svec!(H[U_idxs, v_idx], tempd1d1, cone.rt2)

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::MatrixEpiPerSquare,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    U_idxs = cone.U_idxs
    v_idx = cone.v_idx
    W_idxs = cone.W_idxs
    v2 = 2 * cone.point[cone.v_idx]
    ZiUZi = cone.ZiUZi
    tempd1d1 = Hermitian(cone.tempd1d1, :U)
    temp_U = Hermitian(cone.tempd1d1b, :U)
    temp_U2 = Hermitian(cone.tempd1d1c, :U)
    temp_U3 = cone.tempd1d1d
    temp_W = cone.tempd1d2

    @inbounds for i in 1:size(arr, 2)
        @views svec_to_smat!(temp_U.data, arr[U_idxs, i], cone.rt2)
        @views vec_copyto!(temp_W, arr[W_idxs, i])
        v_arr = arr[v_idx, i]
        @views U_prod = prod[U_idxs, i]
        @views W_prod = prod[W_idxs, i]

        ldiv!(cone.fact_Z, temp_W)
        mul!(temp_U3, temp_W, cone.ZiW')
        copyto!(tempd1d1.data, temp_U)
        rdiv!(ldiv!(cone.fact_Z, tempd1d1.data), cone.fact_Z)
        @. temp_U2.data = temp_U3 + temp_U3' - v2 * tempd1d1.data

        copyto!(tempd1d1, temp_U2)
        axpy!(-2 * v_arr, ZiUZi.data, tempd1d1.data)
        vec_copyto!(W_prod, mul!(temp_W, tempd1d1, cone.W, 2, 2))

        copyto!(tempd1d1, ZiUZi)
        axpby!(-2, cone.Zi.data, 2 * v2, tempd1d1.data)
        prod[v_idx, i] = real(dot(tempd1d1, temp_U)) - 4 *
            real(dot(cone.U, temp_U3)) + cone.Hvv * v_arr
        axpby!(-v2, temp_U2.data, v_arr, tempd1d1.data)
        smat_to_svec!(U_prod, tempd1d1, cone.rt2)
    end

    return prod
end

# TODO reduce allocs
function dder3(cone::MatrixEpiPerSquare, dir::AbstractVector)
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    d2 = cone.d2
    dim = cone.dim
    fact_Z = cone.fact_Z

    U_idxs = cone.U_idxs
    v_idx = cone.v_idx
    W_idxs = cone.W_idxs
    U = cone.U
    v = cone.point[cone.v_idx]
    W = cone.W

    @views U_dir = Hermitian(svec_to_smat!(zero(U.data), dir[U_idxs], cone.rt2))
    v_dir = dir[v_idx]
    @views W_dir = vec_copyto!(zero(W), dir[W_idxs])

    dder3 = cone.dder3
    U_dder3 = view(dder3, U_idxs)
    W_dder3 = view(dder3, W_idxs)

    v2 = 2 * v
    vd2 = 2 * v_dir
    Zi = cone.Zi
    ZiW = cone.ZiW # TODO AKA tau
    ZiU = fact_Z \ U # TODO cache
    ZiWd = fact_Z \ W_dir
    ZiUd = fact_Z \ U_dir
    ZiUZi = cone.ZiUZi
    ZiUZiUZi = Hermitian(ZiUZi * ZiU', :U)
    ZiUZi2v = Hermitian(ZiUZi - v2 * ZiUZiUZi, :U)
    WdWZi = W_dir * ZiW'
    WdZiW = W_dir' * ZiW
    UdZiW = U_dir * ZiW
    ZiWdWZi = fact_Z \ WdWZi
    ZiWdWZi2 = Hermitian(ZiWdWZi + ZiWdWZi', :U)
    ZiUdZiW = fact_Z \ UdZiW
    ZiUZiWdWZi = ZiU * ZiWdWZi
    ZiUZiUdZiW = ZiU * ZiUdZiW + ZiUd * cone.ZiUZiW
    ZiWdWZiUZi = ZiWdWZi * ZiU'
    ZiWdWZiUZi2 = ZiWdWZiUZi + ZiWdWZiUZi' + ZiUZiWdWZi'
    ZiUdZi = Hermitian(ZiUd / fact_Z, :U)
    ZiUZiUdZi = ZiU * ZiUdZi
    ZiUZiUdZi2 = Hermitian(ZiUZiUdZi + ZiUZiUdZi')
    ZiUdZiWdWZi = ZiUd * ZiWdWZi + ZiWdWZi * ZiUd'
    WtZiWI = cone.WtZiW + I
    ZiWdWtZiWI = ZiWd * WtZiWI
    vdZiUZiUZiW = vd2 * ZiUZiUZi * W
    WdWtZiWI = W_dir * WtZiWI
    ZiUZiWdWZiWI = ZiUZi * WdWtZiWI + ZiWdWZiUZi2 * W
    vZiUZiUdZi2 = v * ZiUZiUdZi2 - ZiUdZi

    Utemp = vd2 * (-vd2 * ZiUZi2v + ZiWdWZi2 - v2 * (ZiUZiWdWZi + ZiWdWZiUZi2 -
        2 * vZiUZiUdZi2)) + v2 * (ZiWdWZi * WdWZi + WdWZi' * ZiWdWZi2 +
        ZiWdWtZiWI * ZiWd' + v2 * (v2 * ZiUd * ZiUdZi - ZiUdZiWdWZi - ZiUdZiWdWZi'))
    smat_to_svec!(U_dder3, Utemp, cone.rt2)

    v_Wd_dot = -4 * (v * ZiUZiUdZiW + vdZiUZiUZiW) + ZiUZiWdWZiWI + 2 * ZiUdZiW
    dder3[v_idx] = v_dir * (-8 * dot(ZiUZi2v, U_dir) + v_dir *
        (8 * real(dot(ZiUZiUZi, U)) - (d1 - 1) / v / v / v)) +
        4 * v * real(dot(vZiUZiUdZi2, U_dir)) + 2 * real(dot(v_Wd_dot, W_dir))

    Wtemp = 4 * v_dir * (ZiUdZiW - v2 * ZiUZiUdZiW + ZiUZiWdWZiWI - vdZiUZiUZiW) +
        4 * v * (ZiUdZiW * WdZiW + ZiWdWZi * UdZiW + WdWZi' * ZiUdZiW +
        ZiUdZi * WdWtZiWI - v2 * ZiUd * ZiUdZiW) + -2 * (ZiW * WdZiW * WdZiW +
        WdWZi' * ZiWdWtZiWI + ZiWdWtZiWI * WdZiW + ZiWd * WdZiW' * WtZiWI)
    vec_copyto!(W_dder3, Wtemp)

    return dder3
end
