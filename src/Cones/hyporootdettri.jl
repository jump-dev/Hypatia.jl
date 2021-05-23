"""
$(TYPEDEF)

Hypograph of real symmetric or complex Hermitian root-determinant cone of
dimension `dim` in svec format.

    $(FUNCTIONNAME){T, R}(dim::Int, use_dual::Bool = false)
"""
mutable struct HypoRootdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
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
    hess_fact_cache

    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}
    fact_W
    di::T
    ϕ::T
    ζ::T
    ζi::T
    ϕζidi::T
    dot_const::T
    W::Matrix{R}
    Wi::Matrix{R}
    Wi_vec::Vector{T}
    tempw::Vector{T}

    function HypoRootdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        cone.is_complex = (R <: Complex)
        cone.d = svec_side(R, dim - 1)
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data!(
    cone::HypoRootdetTri{T, R},
    ) where {R <: RealOrComplex{T}} where {T <: Real}
    dim = cone.dim


    # don't alloc here
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)



    d = cone.d
    cone.mat = zeros(R, d, d)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.mat4 = zeros(R, d, d)
    cone.W = zeros(R, d, d)
    cone.Wi_vec = zeros(T, dim - 1)
    cone.tempw = zeros(T, dim - 1)
    cone.di = inv(T(d))
    return cone
end

get_nu(cone::HypoRootdetTri) = (cone.d + 1)

function set_initial_point!(
    arr::AbstractVector{T},
    cone::HypoRootdetTri{T, R},
    ) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    d = cone.d
    const1 = sqrt(T(5d^2 + 2d + 1))
    const2 = arr[1] = -sqrt((3d + 1 - const1) / T(2d + 2))
    const3 = -const2 * (d + 1 + const1) / 2d
    incr = (cone.is_complex ? 2 : 1)
    k = 2
    @inbounds for i in 1:d
        arr[k] = const3
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoRootdetTri{T}) where T
    @assert !cone.feas_updated

    @views svec_to_smat!(cone.mat, cone.point[2:end], cone.rt2)
    fact = cone.fact_W = cholesky!(Hermitian(cone.mat, :U), check = false)
    if isposdef(fact)
        cone.ϕ = exp(logdet(fact) / cone.d)
        cone.ζ = cone.ϕ - cone.point[1]
        cone.is_feas = (cone.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoRootdetTri{T}) where T
    u = cone.dual_point[1]
    if u < -eps(T)
        @views svec_to_smat!(cone.mat2, cone.dual_point[2:end], cone.rt2)
        fact = cholesky!(Hermitian(cone.mat2, :U), check = false)
        return isposdef(fact) && (logdet(fact) -
            cone.d * log(-u / cone.d) > eps(T))
    end
    return false
end

function update_grad(
    cone::HypoRootdetTri{T, R},
    ) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    g = cone.grad
    ζi = cone.ζi = inv(cone.ζ)
    di = cone.di

    g[1] = ζi
    # TODO in-place
    # copyto!(cone.Wi, cone.fact_W.factors)
    # LinearAlgebra.inv!(Cholesky(cone.Wi, 'U', 0)) # TODO inplace for bigfloat
    cone.Wi = inv(cone.fact_W)
    smat_to_svec!(cone.Wi_vec, cone.Wi, cone.rt2)
    ϕζidi = cone.ϕζidi = cone.ϕ * ζi * di
    @. @views g[2:end] = -(ϕζidi + 1) * cone.Wi_vec
    cone.dot_const = ϕζidi * (ϕζidi - di)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoRootdetTri)
    @assert cone.grad_updated
    H = cone.hess.data
    Wi_vec = cone.Wi_vec
    ζi = cone.ζi
    ϕζidi = cone.ϕζidi
    Huwconst = -ϕζidi * ζi
    sckron = ϕζidi + 1
    dot_const = cone.dot_const


    # @warn("don't store it in hessian")

    @inbounds begin
        H[1, 1] = abs2(ζi)
        @. @views H[1, 2:end] = Huwconst * Wi_vec
    end

    @inbounds @views symm_kron!(H[2:end, 2:end], cone.Wi, cone.rt2)
    @inbounds for j in eachindex(Wi_vec)
        j1 = 1 + j
        Wi_vecj = dot_const * Wi_vec[j]
        for i in 1:j
            H[1 + i, j1] = sckron * H[1 + i, j1] + Wi_vec[i] * Wi_vecj
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoRootdetTri,
    )
    @assert cone.grad_updated
    ϕ = cone.ϕ
    ζi = cone.ζi
    di = cone.di
    Wi = Hermitian(cone.Wi, :U)
    w_aux = cone.mat3

    @inbounds @views for j in 1:size(arr, 2)
        p = arr[1, j]
        r = arr[2:end, j]
        r_X = copytri!(svec_to_smat!(cone.mat2, r, cone.rt2), 'U', true)
        c0 = dot(Wi, Hermitian(r_X, :U)) * di
        c1 = -ζi * (p - ϕ * c0) * ζi
        prod[1, j] = -c1
        rwi = rdiv!(r_X, cone.fact_W)
        wirwi = ldiv!(cone.fact_W, r_X)
        @. w_aux = ϕ * (c1 * Wi - ζi * (c0 * Wi - wirwi)) * di + wirwi
        smat_to_svec!(prod[2:end, j], w_aux, cone.rt2)
    end

    return prod
end

function update_inv_hess(cone::HypoRootdetTri)
    @views w = cone.point[2:end]
    svec_to_smat!(cone.W, w, cone.rt2)
    W = Hermitian(cone.W, :U)
    Hi = cone.inv_hess.data
    ζ = cone.ζ
    d = cone.d
    ϕ = cone.ϕ
    den = d * ζ + ϕ
    scdot = ϕ / (d * den)
    sckron = ζ * d / den

    Hi[1, 1] = abs2(ζ) + abs2(ϕ) / d
    Hi12const = ϕ / d
    @. @views Hi[1, 2:end] = Hi12const * w

    @inbounds @views symm_kron!(Hi[2:end, 2:end], W, cone.rt2)
    @inbounds for j in eachindex(w)
        j1 = 1 + j
        scwj = scdot * w[j]
        for i in 1:j
            Hi[1 + i, j1] = sckron * Hi[1 + i, j1] + w[i] * scwj
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoRootdetTri,
    )
    @views w = cone.point[2:end]
    svec_to_smat!(cone.W, w, cone.rt2)
    W = Hermitian(cone.W, :U)
    ζ = cone.ζ
    d = cone.d
    ϕ = cone.ϕ
    const0 = d * ζ + ϕ
    const1 = ϕ / d
    const2 = const1 / const0
    const3 = d * ζ / const0
    const4 = abs2(ζ) + const1 * ϕ

    @inbounds for i in 1:size(arr, 2)
        p = arr[1, i]
        @views r = arr[2:end, i]
        @views prod_w = prod[2:end, i]
        svec_to_smat!(cone.mat2, r, cone.rt2)
        copytri!(cone.mat2, 'U', true)
        mul!(cone.mat3, cone.mat2, W)
        mul!(cone.mat2, W, cone.mat3)
        smat_to_svec!(prod_w, cone.mat2, cone.rt2)
        @. prod_w *= const3
        dot_i = dot(w, r)
        const_i = const2 * dot_i + const1 * p
        @. prod_w += const_i * w
        prod[1, i] = const1 * dot_i + const4 * p
    end

    return prod
end

function dder3(cone::HypoRootdetTri{T}, dir::AbstractVector{T}) where T
    @assert cone.grad_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    dder3 = cone.dder3
    p = dir[1]
    @views r = dir[2:end]
    ϕ = cone.ϕ
    ζi = cone.ζi
    di = cone.di
    w_aux = cone.mat4

    Wi = Hermitian(cone.Wi, :U)
    r_X = copytri!(svec_to_smat!(cone.mat2, r, cone.rt2), 'U', true)
    c0 = dot(Wi, Hermitian(r_X, :U)) * di

    rwi = rdiv!(r_X, cone.fact_W)
    rwi_sqr = real(dot(rwi, rwi')) * di
    L_rwi = ldiv!(cone.fact_W.U', rwi)
    wirwirwi = mul!(cone.mat3, L_rwi', L_rwi)
    wirwi = ldiv!(cone.fact_W.U, L_rwi)

    ζiχ = ζi * (p - ϕ * c0)
    ξbξ = ζi * ϕ * (c0^2 - rwi_sqr) / 2
    c1 = -ζi * (ζiχ^2 - ξbξ)

    c2 = -ζi / 2
    # ∇2h[r] = ϕ * (c0 - rwi) / w * di
    @. w_aux = ζi * ϕ * (c0 * Wi - wirwi) * di
    w_aux .*= ζiχ
    # add c2 * ∇3h[r, r]
    @. w_aux -= c2 * ϕ * ((c0^2 - rwi_sqr) * Wi + 2 * (wirwirwi - c0 * wirwi)) * di

    dder3[1] = c1
    w_aux += wirwirwi - c1 * ϕ * di * Wi
    @views smat_to_svec!(dder3[2:end], w_aux, cone.rt2)

    return dder3
end
