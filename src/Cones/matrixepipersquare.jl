#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

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
    fact_Z::Cholesky{R, Matrix{R}}
    Zi::Hermitian{R, Matrix{R}}
    LiW::Matrix{R}
    ZiW::Matrix{R}
    tempd2d2::Matrix{R}
    tempd1d1::Matrix{R}
    tempd1d1b::Matrix{R}
    tempd1d1c::Matrix{R}
    tempd1d2::Matrix{R}
    tempd1d2b::Matrix{R}
    tempd1d2c::Matrix{R}

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
        cone.W_idxs = (cone.v_idx + 1):(cone.dim)
        return cone
    end
end

function setup_extra_data!(
    cone::MatrixEpiPerSquare{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    (d1, d2) = (cone.d1, cone.d2)
    cone.U = Hermitian(zeros(R, d1, d1), :U)
    cone.W = zeros(R, d1, d2)
    cone.Z = Hermitian(zeros(R, d1, d1), :U)
    cone.Zi = Hermitian(zeros(R, d1, d1), :U)
    cone.LiW = zeros(R, d1, d2)
    cone.ZiW = zeros(R, d1, d2)
    cone.tempd2d2 = zeros(R, d2, d2)
    cone.tempd1d1 = zeros(R, d1, d1)
    cone.tempd1d1b = zeros(R, d1, d1)
    cone.tempd1d1c = zeros(R, d1, d1)
    cone.tempd1d2 = zeros(R, d1, d2)
    cone.tempd1d2b = zeros(R, d1, d2)
    cone.tempd1d2c = zeros(R, d1, d2)
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
    @inbounds for i in 1:(cone.d1)
        arr[k] = 1
        k += incr * i + 1
    end
    arr[cone.v_idx] = 1
    return arr
end

function update_feas(cone::MatrixEpiPerSquare{T}) where {T}
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

function is_dual_feas(cone::MatrixEpiPerSquare{T}) where {T}
    v = cone.dual_point[cone.v_idx]
    if v > eps(T)
        @views svec_to_smat!(cone.tempd1d1b, cone.dual_point[cone.U_idxs], cone.rt2)
        F = cholesky!(Hermitian(cone.tempd1d1b, :U), check = false)
        isposdef(F) || return false
        @views W = vec_copyto!(cone.tempd1d2, cone.dual_point[cone.W_idxs])
        LW = ldiv_tri!(F.U', W)
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
    ldiv_tri!(cone.fact_Z.U', copyto!(cone.LiW, W))
    ldiv_tri!(cone.fact_Z.U, copyto!(cone.ZiW, cone.LiW))
    @views vec_copyto!(cone.grad[cone.W_idxs], cone.ZiW)
    @. @views cone.grad[cone.W_idxs] *= 2

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::MatrixEpiPerSquare)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    d1 = cone.d1
    d2 = cone.d2
    U_idxs = cone.U_idxs
    v_idx = cone.v_idx
    W_idxs = cone.W_idxs
    v = cone.point[v_idx]
    tempd2d2 = cone.tempd2d2
    tempd1d1 = cone.tempd1d1
    Zi = cone.Zi
    ZiW = cone.ZiW
    idx_incr = (cone.is_complex ? 2 : 1)

    # H_v_v part; the v parts substitute U = (Z + W Wᴴ) / (2v)
    mul!(tempd2d2, cone.W', ZiW)
    H[v_idx, v_idx] = (1 + 2 * real(tr(tempd2d2)) + sum(abs2, tempd2d2)) / v / v

    # H_W_W part
    @views tempd2d2[diagind(tempd2d2)] .+= 1

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
                    H[row_idx + 1, col_idx] = imag(term1) - imag(term2)
                    H[row_idx + 1, col_idx + 1] = real(term2) - real(term1)
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
    @views vec_copyto!(H[v_idx, W_idxs], mul!(cone.tempd1d2, ZiW, tempd2d2, -2 / v, false))

    # H_U_v part
    @views smat_to_svec!(H[U_idxs, v_idx], mul!(tempd1d1, ZiW, ZiW', 2, false), cone.rt2)

    cone.hess_updated = true
    return cone.hess
end

# build the inverse Hessian from its closed form, since factorizing the Hessian loses
# accuracy near the boundary
function update_inv_hess(cone::MatrixEpiPerSquare{T}) where {T}
    @assert cone.is_feas
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    Hi = cone.inv_hess.data
    e = zeros(T, cone.dim)
    for j in 1:(cone.dim)
        e[j] = 1
        @views inv_hess_prod!(Hi[:, j], e, cone)
        e[j] = 0
    end
    cone.inv_hess_updated = true
    return cone.inv_hess
end

# solving the Hessian product below for the direction gives, with
# B = Z (arr_W + arr_U W / v) / 2 + v_prod / (2v) * W:
# U: (Z arr_U Z / (2v) + B Wᴴ + W Bᴴ) / (2v), W: B + v_prod / (2v) * W,
# v: v² arr_v + tr(Wᴴ arr_U W) / 2 + v ⟨W, arr_W⟩
function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::MatrixEpiPerSquare{T},
) where {T}
    @assert cone.is_feas
    v = cone.point[cone.v_idx]
    W = cone.W
    fact_U = cone.fact_Z.U
    arr_U = cone.tempd1d1
    BWt = cone.tempd1d1b
    B = cone.tempd1d2
    temp_d1d2 = cone.tempd1d2b

    @inbounds for i in 1:size(arr, 2)
        @views svec_to_smat!(arr_U, arr[cone.U_idxs, i], cone.rt2)
        copytri!(arr_U, 'U', true)
        @views vec_copyto!(B, arr[cone.W_idxs, i])
        mul!(temp_d1d2, Hermitian(arr_U, :U), W)
        v_prod =
            abs2(v) * arr[cone.v_idx, i] + real(dot(W, temp_d1d2)) / 2 + v * real(dot(W, B))
        prod[cone.v_idx, i] = v_prod

        axpy!(inv(v), temp_d1d2, B)
        lmul!(fact_U', lmul!(fact_U, B))
        axpby!(v_prod / (2 * v), W, inv(T(2)), B)
        copyto!(temp_d1d2, B)
        axpy!(v_prod / (2 * v), W, temp_d1d2)
        @views vec_copyto!(prod[cone.W_idxs, i], temp_d1d2)

        lmul!(fact_U, arr_U)
        rmul!(arr_U, fact_U')
        lmul!(fact_U', arr_U)
        rmul!(arr_U, fact_U)
        # reading only one triangle turns the skew part of the rounding error into an
        # error up to cond(Z) times larger in the local norm
        hermitianpart!(arr_U)
        mul!(BWt, B, W')
        @. arr_U = (arr_U / (2 * v) + BWt + BWt') / (2 * v)
        @views smat_to_svec!(prod[cone.U_idxs, i], arr_U, cone.rt2)
    end

    return prod
end

#=
f = -logdet(X) + (d2 - 1) log(v) + const, where X = [U W; Wᴴ 2vI] is linear in the point.
With Z = L Lᴴ, X⁻¹ = K Kᴴ for K = [√(2v) L⁻ᴴ 0; -Wᴴ L⁻ᴴ / √(2v) I / √(2v)], and the
direction whitened by K is S = Kᴴ DX[dir] K = [S11 Y; Yᴴ v_rel I], where v_rel = v_dir / v,
S11 = L⁻¹ (2v U_dir - H Wᴴ - W Hᴴ) L⁻ᴴ with H = W_dir - v_rel / 2 * W,
and Y = L⁻¹ (W_dir - v_rel W). The Hessian product and the third order term of -logdet(X)
are the adjoint of the map to X applied to K S Kᴴ and to (K S)(K S)ᴴ. This avoids the
cancellation in the derivatives of Z near the boundary.
=#

# computes S11, Y and E = Y - S11 L⁻¹ W for the direction, and returns v_rel
function whiten_dir!(cone::MatrixEpiPerSquare, dir::AbstractVector)
    v = cone.point[cone.v_idx]
    W = cone.W
    fact_U = cone.fact_Z.U
    S11 = cone.tempd1d1
    temp_d1d1 = cone.tempd1d1c
    Y = cone.tempd1d2
    E = cone.tempd1d2b

    @views svec_to_smat!(S11, dir[cone.U_idxs], cone.rt2)
    v_rel = dir[cone.v_idx] / v
    @views vec_copyto!(Y, dir[cone.W_idxs])

    axpy!(-v_rel / 2, W, Y)
    mul!(temp_d1d1, Y, W')
    @. S11 = 2 * v * S11 - temp_d1d1 - temp_d1d1'
    copytri!(S11, 'U', true)
    rdiv!(ldiv_tri!(fact_U', S11), fact_U)
    hermitianpart!(S11)

    ldiv_tri!(fact_U', Y)
    axpy!(-v_rel / 2, cone.LiW, Y)
    mul!(E, Hermitian(S11, :U), cone.LiW)
    axpby!(true, Y, -1, E)
    return v_rel
end

# U: 2v L⁻ᴴ S11 L⁻¹, v: (v_rel - Re⟨L⁻¹ W, E⟩ - Re⟨L⁻¹ W, Y⟩) / v, W: 2 L⁻ᴴ E
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::MatrixEpiPerSquare)
    @assert cone.grad_updated
    v = cone.point[cone.v_idx]
    LiW = cone.LiW
    fact_U = cone.fact_Z.U
    S11 = cone.tempd1d1
    Y = cone.tempd1d2
    E = cone.tempd1d2b

    @inbounds for i in 1:size(arr, 2)
        v_rel = @views whiten_dir!(cone, arr[:, i])
        prod[cone.v_idx, i] = (v_rel - real(dot(LiW, E)) - real(dot(LiW, Y))) / v
        rdiv!(ldiv_tri!(fact_U, S11), fact_U')
        @views smat_to_svec!(prod[cone.U_idxs, i], S11, cone.rt2)
        @views prod[cone.U_idxs, i] .*= 2 * v
        ldiv_tri!(fact_U, E)
        @views vec_copyto!(prod[cone.W_idxs, i], E)
        @views prod[cone.W_idxs, i] .*= 2
    end

    return prod
end

function dder3(cone::MatrixEpiPerSquare, dir::AbstractVector)
    @assert cone.grad_updated
    v = cone.point[cone.v_idx]
    LiW = cone.LiW
    fact_U = cone.fact_Z.U
    S11 = cone.tempd1d1
    A11 = cone.tempd1d1b
    temp_d1d1 = cone.tempd1d1c
    Y = cone.tempd1d2
    E = cone.tempd1d2b
    A12 = cone.tempd1d2c
    dder3 = cone.dder3

    v_rel = whiten_dir!(cone, dir)
    ldiv_tri!(fact_U, copyto!(A11, S11))
    ldiv_tri!(fact_U, copyto!(A12, Y))

    # U: 2v (A11 A11ᴴ + A12 A12ᴴ), with A11 = L⁻ᴴ S11 and A12 = L⁻ᴴ Y
    mul!(temp_d1d1, A11, A11')
    mul!(temp_d1d1, A12, A12', true, true)
    @views smat_to_svec!(dder3[cone.U_idxs], temp_d1d1, cone.rt2)
    @views dder3[cone.U_idxs] .*= 2 * v

    # v: (‖E‖² + ‖v_rel I - Yᴴ L⁻¹ W‖² - (d2 - 1) v_rel²) / v
    mul!(temp_d1d1, Y, Y')
    mul!(S11, LiW, LiW')
    dder3[cone.v_idx] =
        (
            sum(abs2, E) + abs2(v_rel) - 2 * v_rel * real(dot(LiW, Y)) +
            real(dot(temp_d1d1, S11))
        ) / v

    # W: 2 (A11 E + A12 (v_rel I - Yᴴ L⁻¹ W))
    mul!(temp_d1d1, A12, Y')
    mul!(A12, temp_d1d1, LiW, -1, v_rel)
    mul!(A12, A11, E, true, true)
    @views vec_copyto!(dder3[cone.W_idxs], A12)
    @views dder3[cone.W_idxs] .*= 2

    return dder3
end

function pretty_name(cone::MatrixEpiPerSquare)
    realorcomplex = cone.is_complex ? "complex " : "real "
    dualorprimal = use_dual_barrier(cone) ? "dual " : ""
    return realorcomplex * dualorprimal * "matrix outer product"
end
