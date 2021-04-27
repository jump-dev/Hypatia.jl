#=
vector cone of squares, i.e. ℝ₊ᵈ for d ≥ 1, with rank d
=#

struct VectorCSqr{T <: Real} <: ConeOfSquares{T} end

vector_dim(::Type{<:VectorCSqr}, d::Int) = d

mutable struct VectorCSqrCache{T <: Real} <: CSqrCache{T}
    viw::Vector{T}
    wi::Vector{T}
    ϕ::T
    ζ::T
    ζi::T
    σ::T
    ∇h_viw::Vector{T}
    ∇2h_viw::Vector{T}
    ∇3h_viw::Vector{T}
    ζi∇h_viw::Vector{T}
    VectorCSqrCache{T}() where {T <: Real} = new{T}()
end

function setup_csqr_cache(cone::EpiPerSepSpectral{VectorCSqr{T}}) where T
    cone.cache = cache = VectorCSqrCache{T}()
    d = cone.d
    cache.viw = zeros(T, d)
    cache.wi = zeros(T, d)
    cache.∇h_viw = zeros(T, d)
    cache.∇2h_viw = zeros(T, d)
    cache.∇3h_viw = zeros(T, d)
    cache.ζi∇h_viw = zeros(T, d)
    return
end

function set_initial_point(arr::AbstractVector, cone::EpiPerSepSpectral{<:VectorCSqr})
    (arr[1], arr[2], w0) = get_initial_point(cone.d, cone.h)
    @views fill!(arr[3:end], w0)
    return arr
end

function update_feas(cone::EpiPerSepSpectral{VectorCSqr{T}}) where T
    @assert !cone.feas_updated
    cache = cone.cache
    v = cone.point[2]

    if (v > eps(T)) && all(>(eps(T)), cone.w_view)
        @. cache.viw = cone.w_view / v
        cache.ϕ = h_val(cache.viw, cone.h)
        cache.ζ = cone.point[1] - v * cache.ϕ
        cone.is_feas = (cache.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiPerSepSpectral{VectorCSqr{T}}) where T
    u = cone.dual_point[1]
    (u < eps(T)) && return false
    @views w = cone.dual_point[3:end]
    h_conj_dom_pos(cone.h) && any(<(eps(T)), w) && return false

    # TODO in-place:
    temp = similar(w)
    @. temp = w / u
    # h_conj_dom(temp, cone.h) || return false
    return (cone.dual_point[2] - u * h_conj(temp, cone.h) > eps(T))
end

function update_grad(cone::EpiPerSepSpectral{VectorCSqr{T}}) where T
    @assert !cone.grad_updated && cone.is_feas
    grad = cone.grad
    v = cone.point[2]
    cache = cone.cache
    ζi = cache.ζi = inv(cache.ζ)
    ∇h_viw = cache.∇h_viw
    h_der1(∇h_viw, cache.viw, cone.h)
    cache.σ = cache.ϕ - dot(cache.viw, ∇h_viw)
    @. cache.wi = inv(cone.w_view)
    @. cache.ζi∇h_viw = ζi * ∇h_viw

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @. grad[3:end] = -cache.wi + cache.ζi∇h_viw

    cone.grad_updated = true
    return grad
end

function update_hess(cone::EpiPerSepSpectral{VectorCSqr{T}}) where T
    @assert cone.grad_updated && !cone.hess_updated
    d = cone.d
    v = cone.point[2]
    cache = cone.cache
    H = cone.hess.data
    ζi = cache.ζi
    ζi2 = abs2(ζi)
    σ = cache.σ
    viw = cache.viw
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    h_der2(∇2h_viw, cache.viw, cone.h)
    ζi∇h_viw = cache.ζi∇h_viw
    wi = cache.wi
    ζivi = ζi / v
    ζiσ = ζi * σ

    # Huu
    H[1, 1] = ζi2

    # Huv
    H[1, 2] = -ζi2 * σ

    # Hvv start
    Hvv = v^-2 + abs2(ζi * σ)

    @inbounds for j in 1:d
        ζi∇h_viw_j = ζi∇h_viw[j]
        term_j = ζivi * viw[j] * ∇2h_viw[j]
        Hvv += viw[j] * term_j
        j2 = 2 + j

        # Huw
        H[1, j2] = -ζi * ζi∇h_viw_j

        # Hvw
        H[2, j2] = ζiσ * ζi∇h_viw_j - term_j

        # Hww
        for i in 1:(j - 1)
            H[2 + i, j2] = ζi∇h_viw_j * ζi∇h_viw[i]
        end
        H[j2, j2] = abs2(ζi∇h_viw_j) + ζivi * ∇2h_viw[j] + abs2(wi[j])
    end

    # Hvv end
    H[2, 2] = Hvv

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSepSpectral{VectorCSqr{T}}) where T
    # cone.hess_aux_updated || update_hess_aux(cone) # TODO
    hess(cone) # TODO
    @assert cone.hess_updated
    v = cone.point[2]
    w = cone.w_view
    cache = cone.cache
    ζi = cache.ζi
    ζi2 = abs2(ζi)
    viw = cache.viw
    σ = cache.σ
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    wi = cache.wi

    # TODO prealloc
    d = cone.d
    ξb = zeros(d)
    ζivi = ζi / v

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views r = arr[3:end, j]

        viq = q / v
        @. ξb = ζivi * ∇2h_viw * (-viq * w + r)
        χ = p - σ * q - dot(∇h_viw, r)
        ζi2χ = ζi2 * χ

        prod[1, j] = ζi2χ
        prod[2, j] = -ζi2χ * σ - dot(viw, ξb) + viq / v
        @. prod[3:end, j] = -ζi2χ * ∇h_viw + ξb + r * wi * wi
    end

    return prod
end

function update_inv_hess(cone::EpiPerSepSpectral{VectorCSqr{T}}) where T
    hess(cone) # TODO
    @assert cone.hess_updated # TODO
    d = cone.d
    v = cone.point[2]
    cache = cone.cache
    Hi = cone.inv_hess.data
    ζi = cache.ζi
    ζ = cache.ζ
    ζi2 = abs2(ζi)
    σ = cache.σ
    viw = cache.viw
    w = cone.w_view
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    wi = cache.wi
    ζivi = ζi / v

    # TODO in-place
    m = inv.(ζivi * ∇2h_viw + abs2.(wi))
    α = m .* ∇h_viw
    w∇2h_viw = ζivi * ∇2h_viw .* viw
    γ = m .* w∇2h_viw

    ζ2β = abs2(ζ) + dot(∇h_viw, α)
    c0 = σ + dot(∇h_viw, γ)
    c1 = c0 / ζ2β
    c3 = abs2(inv(v)) + σ * c1 + sum((viw[i] + c1 * α[i] - γ[i]) * w∇2h_viw[i] for i in 1:d)
    c4 = inv(c3 - c0 * c1)

    # Hiuu, Hiuv, Hivv
    Hi[1, 1] = c4 * ζ2β * c3
    Hi[1, 2] = c4 * c0
    Hi[2, 2] = c4

    @inbounds for j in 1:d
        j2 = 2 + j

        # Hivw
        Hivj = Hi[2, j2] = c4 * γ[j]

        # Hiuw
        Hi[1, j2] = α[j] + c0 * Hivj

        # Hiww
        for i in 1:j
            Hi[2 + i, j2] = Hivj * γ[i]
        end
        Hi[j2, j2] += m[j]
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSepSpectral{VectorCSqr{T}}) where T
    # cone.hess_aux_updated || update_hess_aux(cone) # TODO
    hess(cone) # TODO
    @assert cone.hess_updated # TODO
    d = cone.d
    v = cone.point[2]
    cache = cone.cache
    Hi = cone.inv_hess.data
    ζi = cache.ζi
    ζ = cache.ζ
    ζi2 = abs2(ζi)
    σ = cache.σ
    viw = cache.viw
    w = cone.w_view
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    ζi∇h_viw = cache.ζi∇h_viw
    wi = cache.wi
    ζivi = ζi / v

    # TODO in-place
    m = inv.(ζivi * ∇2h_viw + abs2.(wi))
    α = m .* ∇h_viw
    w∇2h_viw = ζivi * ∇2h_viw .* viw
    γ = m .* w∇2h_viw

    ζ2β = abs2(ζ) + dot(∇h_viw, α)
    c0 = σ + dot(∇h_viw, γ)
    c1 = c0 / ζ2β
    c3 = abs2(inv(v)) + σ * c1 + sum((viw[i] + c1 * α[i] - γ[i]) * w∇2h_viw[i] for i in 1:d)
    c4 = inv(c3 - c0 * c1)

    c5 = ζ2β * c3

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views r = arr[3:end, j]

        qγr = q + dot(γ, r)
        cu = c4 * (c5 * p + c0 * qγr)
        cv = c4 * (c0 * p + qγr)

        prod[1, j] = cu + dot(α, r)
        prod[2, j] = cv
        @. prod[3:end, j] = p * α + cv * γ + m * r
    end

    return prod
end

function correction(cone::EpiPerSepSpectral{VectorCSqr{T}}, dir::AbstractVector{T}) where T
    @assert cone.hess_updated
    v = cone.point[2]
    w = cone.w_view
    cache = cone.cache
    ζi = cache.ζi
    viw = cache.viw
    σ = cache.σ
    ∇h_viw = cache.∇h_viw
    ∇2h_viw = cache.∇2h_viw
    ∇3h_viw = cache.∇3h_viw
    h_der3(∇3h_viw, cache.viw, cone.h)
    wi = cache.wi
    corr = cone.correction

    # TODO prealloc
    d = cone.d
    ξ = zeros(d)
    ξb = zeros(d)
    temp = zeros(d)

    p = dir[1]
    q = dir[2]
    @views r = dir[3:end]

    viq = q / v
    @. ξ = -viq * w + r
    ζivi = ζi / v
    @. ξb = ζivi * ∇2h_viw * ξ
    χ = p - σ * q - dot(∇h_viw, r)
    ζiχ = ζi * χ

    ξbξ = dot(ξb, ξ) / 2
    c1 = ζi * (ζiχ^2 + ξbξ)

    ζiχpviq = ζiχ + viq
    c2 = ζi / 2
    @. ξ /= v
    @. temp = ζiχpviq * ξb - c2 * ∇3h_viw .* ξ .* ξ

    corr[1] = c1
    corr[2] = -c1 * σ - dot(viw, temp) + (ξbξ + viq^2) / v
    @. corr[3:end] = -c1 * ∇h_viw + temp + abs2(r * wi) * wi

    return corr
end
