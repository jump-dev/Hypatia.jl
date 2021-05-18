"""
$(TYPEDEF)

Hypograph of perspective function of real symmetric or complex Hermitian
log-determinant cone of dimension `dim` in svec format.

    $(FUNCTIONNAME){T, R}(dim::Int, use_dual::Bool = false)
"""
mutable struct HypoPerLogdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
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
    inv_hess_aux_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}
    mat5::Matrix{R}
    fact_W
    ϕ::T
    σ::T
    ζ::T
    ζi::T
    W::Matrix{R}
    Wi::Matrix{R}
    Wi_vec::Vector{T}
    tempw::Vector{T}
    c0::T
    c4::T
    Hiuu::T

    function HypoPerLogdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 3
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        if R <: Complex
            d = isqrt(dim - 2) # real lower triangle and imaginary under diagonal
            @assert d^2 == dim - 2
            cone.is_complex = true
        else
            d = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
            @assert d * (d + 1) == 2 * (dim - 2)
            cone.is_complex = false
        end
        cone.d = d
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

reset_data(cone::HypoPerLogdetTri) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_aux_updated =
    cone.hess_fact_updated = false)

function setup_extra_data!(
    cone::HypoPerLogdetTri{T, R},
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
    cone.mat5 = zeros(R, d, d)
    cone.W = zeros(R, d, d)
    cone.Wi_vec = zeros(T, dim - 2)
    cone.tempw = zeros(T, dim - 2)
    return cone
end

get_nu(cone::HypoPerLogdetTri) = cone.d + 2

function set_initial_point!(
    arr::AbstractVector{T},
    cone::HypoPerLogdetTri{T, R},
    ) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    # central point data are the same as for hypoperlog
    (arr[1], arr[2], w) = get_central_ray_hypoperlog(cone.d)
    incr = (cone.is_complex ? 2 : 1)
    k = 3
    @inbounds for i in 1:cone.d
        arr[k] = w
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoPerLogdetTri{T}) where T
    @assert !cone.feas_updated
    v = cone.point[2]

    if v > eps(T)
        u = cone.point[1]
        @views svec_to_smat!(cone.mat, cone.point[3:end], cone.rt2)
        fact = cone.fact_W = cholesky!(Hermitian(cone.mat, :U), check = false)
        if isposdef(fact)
            cone.ϕ = logdet(fact) - cone.d * log(v)
            cone.ζ = v * cone.ϕ - u
            cone.is_feas = (cone.ζ > eps(T))
        else
            cone.is_feas = false
        end
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoPerLogdetTri{T}) where T
    u = cone.dual_point[1]
    if u < -eps(T)
        v = cone.dual_point[2]
        @views svec_to_smat!(cone.mat2, cone.dual_point[3:end], cone.rt2)
        fact = cholesky!(Hermitian(cone.mat2, :U), check = false)
        return isposdef(fact) &&
            (v - u * (logdet(fact) + cone.d * (1 - log(-u))) > eps(T))
    end
    return false
end

function update_grad(cone::HypoPerLogdetTri)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    g = cone.grad
    ζ = cone.ζ
    ζi = cone.ζi = inv(ζ)
    cone.σ = cone.ϕ - cone.d

    g[1] = ζi
    g[2] = -inv(v) - cone.σ / ζ
    # TODO in-place
    # copyto!(cone.Wi, cone.fact_W.factors)
    # LinearAlgebra.inv!(Cholesky(cone.Wi, 'U', 0))
    cone.Wi = inv(cone.fact_W)
    smat_to_svec!(cone.Wi_vec, cone.Wi, cone.rt2)
    # ∇ϕ = cone.Wi_vec * v
    zvzi = -(ζ + v) / ζ
    @inbounds @. @views g[3:end] = zvzi * cone.Wi_vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLogdetTri)
    @assert cone.grad_updated
    v = cone.point[2]
    H = cone.hess.data
    ζ = cone.ζ
    ζi = cone.ζi
    σ = cone.σ
    Wi_vec = cone.Wi_vec
    zvzi = (ζ + v) / ζ
    vzi = zvzi - 1
    Wivzi = cone.tempw
    ζi2 = abs2(ζi)
    ζivi = ζi / v
    @. Wivzi = vzi * Wi_vec
    d = cone.d

    # Huu
    H[1, 1] = ζi2

    # Huv
    H[1, 2] = -ζi2 * σ

    # Hvv
    H[2, 2] = v^-2 + abs2(ζi * σ) + ζivi * d

    @inbounds begin
        # Huw
        H13const = -v / ζ / ζ
        @. @views H[1, 3:end] = H13const * Wi_vec

        # Hvw
        H23const = ((cone.ϕ - d) * v / ζ - 1) / ζ
        @. @views H[2, 3:end] = H23const * Wi_vec
    end

    # Hww
    @inbounds @views symm_kron!(H[3:end, 3:end], cone.Wi, cone.rt2)
    @inbounds for j in eachindex(Wi_vec)
        j2 = 2 + j
        Wivzij = Wivzi[j]
        for i in 1:j
            H[2 + i, j2] = zvzi * H[2 + i, j2] + Wivzi[i] * Wivzij
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPerLogdetTri,
    )
    @assert cone.grad_updated
    v = cone.point[2]
    ζ = cone.ζ
    ζi = cone.ζi
    σ = cone.σ
    Wi = Hermitian(cone.Wi, :U)
    w_aux = cone.mat3

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views r_X = svec_to_smat!(cone.mat2, arr[3:end, j], cone.rt2)

        c0 = dot(Wi, Hermitian(r_X, :U))
        ∇ϕr = c0 * v
        c1 = ζi * (-p + σ * q + ∇ϕr) * ζi
        c2 = (q * cone.d / v - c0) / ζ

        copytri!(r_X, 'U', true)
        ldiv!(cone.fact_W, r_X)
        rdiv!(r_X, cone.fact_W)

        # negative tau
        @. w_aux = (v * r_X - q * Wi) / ζ
        @. w_aux += c1 * v * Wi + r_X

        prod[1, j] = -c1
        prod[2, j] = c1 * σ + c2 + q / v / v
        @views smat_to_svec!(prod[3:end, j], w_aux, cone.rt2)
    end

    return prod
end

function update_inv_hess_aux(cone::HypoPerLogdetTri)
    @assert cone.feas_updated
    @assert !cone.inv_hess_aux_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    svec_to_smat!(cone.W, w, cone.rt2)
    Hi = cone.inv_hess.data
    d = cone.d
    ζ = cone.ζ
    ζv = ζ + v
    ζuζ = 2 * ζ + u
    den = ζv + d * v

    c0 = cone.ϕ - d * ζ / (ζ + v)
    Hiuu = abs2(ζ + u) + ζ * (den - v) - d * abs2(ζuζ) * v / den
    c4 = v^2 / den * ζv
    cone.c0 = c0
    cone.c4 = c4
    cone.Hiuu = Hiuu

    cone.inv_hess_aux_updated = true
    return
end

function update_inv_hess(cone::HypoPerLogdetTri)
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    v = cone.point[2]
    @views w = cone.point[3:end]
    W = Hermitian(cone.W, :U)
    Hi = cone.inv_hess.data
    ζ = cone.ζ
    ζv = ζ + v
    ζζv = ζ / ζv

    γ_vec = cone.tempw
    @. γ_vec = w / ζv

    c0 = cone.c0
    c4 = cone.c4
    Hi[1, 1] = cone.Hiuu
    Hi[1, 2] = c0 * c4
    Hi[2, 2] = c4
    α_const = v * ζζv
    @inbounds @views begin
        @. Hi[1, 3:end] = (α_const + c0 * c4 / ζv) * w
        @. Hi[2, 3:end] = c4 * w / ζv
        symm_kron!(Hi[3:end, 3:end], W, cone.rt2)
        @. Hi[3:end, 3:end] *= ζζv
        mul!(Hi[3:end, 3:end], γ_vec, γ_vec', c4, true)
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPerLogdetTri,
    )
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    v = cone.point[2]
    W = Hermitian(cone.W, :U)
    ζ = cone.ζ
    ζv = ζ + v
    ζi = cone.ζi
    c0 = cone.c0
    c4 = cone.c4
    α_const = v / (v * ζi + 1)
    w_aux = cone.mat3
    w_prod = cone.mat4
    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views r = arr[3:end, j]
        @views R = svec_to_smat!(cone.mat4, arr[3:end, j], cone.rt2)
        copytri!(R, 'U', true)
        @views prod_w = prod[3:end, j]
        rw_const = dot(Hermitian(R, :U), W)
        qγr = q + rw_const / ζv
        cv = c4 * (c0 * p + qγr)
        prod[1, j] = cone.Hiuu * p + c4 * c0 * qγr + rw_const * α_const
        prod[2, j] = cv
        mul!(w_aux, W, R)
        mul!(w_prod, w_aux, W, ζ / ζv, false)
        @. w_prod += (p * α_const + cv / ζv) * W
        @views smat_to_svec!(prod[3:end, j], w_prod, cone.rt2)
    end

    return prod
end

function dder3(cone::HypoPerLogdetTri, dir::AbstractVector)
    @assert cone.grad_updated
    p = dir[1]
    q = dir[2]
    @views r = dir[3:end]
    dder3 = cone.dder3
    v = cone.point[2]
    ζ = cone.ζ
    ζi = cone.ζi
    σ = cone.σ
    d = cone.d
    Wi = Hermitian(cone.Wi)
    @views w = cone.point[3:end]
    W = Hermitian(svec_to_smat!(cone.W, w, cone.rt2), :U)
    viq = q / v
    viq2 = abs2(viq)
    w_aux = cone.mat5

    r_X = copytri!(svec_to_smat!(cone.mat3, r, cone.rt2), 'U', true)
    c0 = dot(Wi, r_X)
    ∇ϕr = c0 * v
    χ = -p + σ * q + ∇ϕr

    rwi = rdiv!(r_X, cone.fact_W)
    rwi_sqr = real(dot(rwi, rwi'))
    L_rwi = ldiv!(cone.fact_W.U', rwi)
    wirwirwi = mul!(cone.mat4, L_rwi', L_rwi)
    wirwi = ldiv!(cone.fact_W.U, L_rwi)

    # tau of the same form as prod
    @. w_aux = (q * Wi - v * wirwi) / ζ
    # tau of TOO
    ζiχ = ζi * χ
    @. w_aux *= -ζiχ - viq
    @. w_aux += v * (viq2 * Wi - 2 * viq * wirwi + wirwirwi) / ζ

    ∇2ϕξξ = -viq2 * d + 2 * viq * c0 - rwi_sqr
    c1 = ζi * (abs2(ζiχ) - v * ∇2ϕξξ * ζi / 2)
    c2 = (viq * d - c0) / ζ

    dder3[1] = -c1
    dder3[2] = c1 * σ + (viq2 - dot(Hermitian(w_aux, :U), W)) / v - ∇2ϕξξ / ζ / 2
    @. w_aux += c1 * v * Wi + wirwirwi
    @views smat_to_svec!(dder3[3:end], w_aux, cone.rt2)

    return dder3
end
