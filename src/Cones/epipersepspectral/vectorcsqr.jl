"""
$(TYPEDEF)

Real vector cone of squares.
"""
struct VectorCSqr{T <: Real} <: ConeOfSquares{T} end

"""
$(TYPEDSIGNATURES)

The rank of the vector cone of squares, equal to the vector length.
"""
vector_dim(::Type{<:VectorCSqr}, d::Int) = d

mutable struct VectorCSqrCache{T <: Real} <: CSqrCache{T}
    viw::Vector{T}
    wi::Vector{T}
    ϕ::T
    ζ::T
    ζi::T
    ζivi::T
    σ::T
    ∇h::Vector{T}
    ∇2h::Vector{T}
    ∇3h::Vector{T}

    m::Vector{T}
    α::Vector{T}
    γ::Vector{T}
    k1::T
    k2::T
    k3::T

    w1::Vector{T}
    w2::Vector{T}

    VectorCSqrCache{T}() where {T <: Real} = new{T}()
end

function setup_csqr_cache(cone::EpiPerSepSpectral{VectorCSqr{T}}) where {T}
    cone.cache = cache = VectorCSqrCache{T}()
    d = cone.d
    cache.viw = zeros(T, d)
    cache.wi = zeros(T, d)
    cache.∇h = zeros(T, d)
    cache.∇2h = zeros(T, d)
    cache.∇3h = zeros(T, d)
    cache.w1 = zeros(T, d)
    cache.w2 = zeros(T, d)
    cache.m = zeros(T, d)
    cache.α = zeros(T, d)
    cache.γ = zeros(T, d)
    return
end

function set_initial_point!(arr::AbstractVector, cone::EpiPerSepSpectral{<:VectorCSqr})
    (arr[1], arr[2], w0) = get_initial_point(cone.d, cone.h)
    @views fill!(arr[3:end], w0)
    return arr
end

function update_feas(cone::EpiPerSepSpectral{VectorCSqr{T}}) where {T}
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

function is_dual_feas(cone::EpiPerSepSpectral{VectorCSqr{T}}) where {T}
    u = cone.dual_point[1]
    (u < eps(T)) && return false
    @views w = cone.dual_point[3:end]

    h_conj_dom_pos(cone.h) && any(<(eps(T)), w) && return false
    uiw = cone.cache.w1
    @. uiw = w / u
    return (cone.dual_point[2] - u * h_conj(uiw, cone.h) > eps(T))
end

function update_grad(cone::EpiPerSepSpectral{<:VectorCSqr})
    @assert !cone.grad_updated && cone.is_feas
    grad = cone.grad
    v = cone.point[2]
    cache = cone.cache
    ζi = cache.ζi = inv(cache.ζ)
    ∇h = cache.∇h
    h_der1(∇h, cache.viw, cone.h)
    cache.σ = cache.ϕ - dot(cache.viw, ∇h)
    @. cache.wi = inv(cone.w_view)

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @. grad[3:end] = -cache.wi + ζi * ∇h

    cone.grad_updated = true
    return grad
end

function update_hess_aux(cone::EpiPerSepSpectral{<:VectorCSqr})
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    cache = cone.cache
    cache.ζivi = inv(cache.ζ * cone.point[2])
    h_der2(cache.∇2h, cache.viw, cone.h)
    return cone.hess_aux_updated = true
end

function update_hess(cone::EpiPerSepSpectral{<:VectorCSqr})
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    v = cone.point[2]
    H = cone.hess.data
    cache = cone.cache
    ζi = cache.ζi
    ζivi = cache.ζivi
    σ = cache.σ
    viw = cache.viw
    wi = cache.wi
    ∇h = cache.∇h
    ∇2h = cache.∇2h

    # Huu
    Huu = H[1, 1] = abs2(ζi)

    # Huv
    H[1, 2] = -Huu * σ

    # Hvv start
    Hvv = v^-2 + abs2(ζi * σ)

    @inbounds for j in 1:(cone.d)
        ζi∇h_j = ζi * ∇h[j]
        ζi2∇h_j = ζi * ζi∇h_j
        ζivi∇2h_j = ζivi * ∇2h[j]
        term_j = viw[j] * ζivi∇2h_j
        Hvv += viw[j] * term_j
        j2 = 2 + j

        # Huw
        Huwj = H[1, j2] = -ζi2∇h_j

        # Hvw
        H[2, j2] = σ * ζi2∇h_j - term_j

        # Hww
        for i in 1:(j - 1)
            H[2 + i, j2] = ζi2∇h_j * ∇h[i]
        end
        H[j2, j2] = abs2(ζi∇h_j) + ζivi∇2h_j + abs2(wi[j])
    end

    # Hvv end
    H[2, 2] = Hvv

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSepSpectral{VectorCSqr{T}},
) where {T}
    cone.hess_aux_updated || update_hess_aux(cone)
    v = cone.point[2]
    w = cone.w_view
    cache = cone.cache
    ζi = cache.ζi
    ζivi = cache.ζivi
    viw = cache.viw
    wi = cache.wi
    σ = cache.σ
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    ξb = cache.w1

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views r = arr[3:end, j]

        viq = q / v
        @. ξb = ζivi * ∇2h * (r - viq * w)
        c1 = -ζi * (p - σ * q - dot(∇h, r)) * ζi

        prod[1, j] = -c1
        prod[2, j] = c1 * σ - dot(viw, ξb) + viq / v
        @. prod[3:end, j] = c1 * ∇h + ξb + wi * r * wi
    end

    return prod
end

function update_inv_hess_aux(cone::EpiPerSepSpectral{<:VectorCSqr})
    @assert !cone.inv_hess_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)
    v = cone.point[2]
    cache = cone.cache
    ∇h = cache.∇h
    wi = cache.wi
    w1 = cache.w1
    m = cache.m
    α = cache.α
    γ = cache.γ

    @. w1 = cache.ζivi * cache.∇2h
    @. m = inv(w1 + abs2(wi))
    @. α = m * ∇h
    @. γ = m * cache.viw * w1

    cache.k1 = abs2(cache.ζ) + dot(∇h, α)
    cache.k2 = cache.σ + dot(∇h, γ)
    cache.k3 = (inv(v) + dot(wi, γ)) / v

    return cone.inv_hess_aux_updated = true
end

function update_inv_hess(cone::EpiPerSepSpectral{<:VectorCSqr})
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    Hi = cone.inv_hess.data
    cache = cone.cache
    m = cache.m
    α = cache.α
    γ = cache.γ
    k2 = cache.k2
    k3 = cache.k3

    # Hiuu, Hiuv, Hivv
    k23i = k2 / k3
    Hi[1, 1] = cache.k1 + k23i * k2
    Hi[1, 2] = k23i
    Hi[2, 2] = inv(k3)

    @inbounds for j in 1:(cone.d)
        j2 = 2 + j

        # Hivw
        Hivj = Hi[2, j2] = γ[j] / k3

        # Hiuw
        Hi[1, j2] = α[j] + k2 * Hivj

        # Hiww
        for i in 1:j
            Hi[2 + i, j2] = Hivj * γ[i]
        end
        Hi[j2, j2] += m[j]
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSepSpectral{VectorCSqr{T}},
) where {T}
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    cache = cone.cache
    m = cache.m
    α = cache.α
    γ = cache.γ
    k1 = cache.k1
    k2 = cache.k2
    k3 = cache.k3

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views r = arr[3:end, j]

        cv = (k2 * p + arr[2, j] + dot(γ, r)) / k3

        prod[1, j] = k1 * p + k2 * cv + dot(α, r)
        prod[2, j] = cv
        @. @views prod[3:end, j] = p * α + cv * γ + m * r
    end

    return prod
end

function update_dder3_aux(cone::EpiPerSepSpectral{<:VectorCSqr})
    @assert !cone.dder3_aux_updated
    cone.hess_aux_updated || update_hess_aux(cone)
    h_der3(cone.cache.∇3h, cone.cache.viw, cone.h)
    return cone.dder3_aux_updated = true
end

function dder3(cone::EpiPerSepSpectral{VectorCSqr{T}}, dir::AbstractVector{T}) where {T}
    cone.dder3_aux_updated || update_dder3_aux(cone)
    v = cone.point[2]
    w = cone.w_view
    dder3 = cone.dder3
    cache = cone.cache
    ζi = cache.ζi
    σ = cache.σ
    ∇h = cache.∇h
    ∇2h = cache.∇2h
    wi = cache.wi
    ξ = cache.w1
    ξb = cache.w2

    p = dir[1]
    q = dir[2]
    @views r = dir[3:end]

    viq = q / v
    @. ξ = r - viq * w
    @. ξb = cache.ζivi * ∇2h * ξ
    ζiχ = ζi * (p - σ * q - dot(∇h, r))
    ξbξ = dot(ξb, ξ) / 2
    c1 = -ζi * (ζiχ^2 + ξbξ)

    c2 = -ζi / 2
    ξ ./= v
    w_aux = ξb
    w_aux .*= ζiχ + viq
    @. w_aux += c2 * ξ * cache.∇3h * ξ

    dder3[1] = -c1
    dder3[2] = c1 * σ - dot(cache.viw, w_aux) + (ξbξ + viq^2) / v
    @. dder3[3:end] = c1 * ∇h + w_aux + abs2(r * wi) * wi

    return dder3
end
