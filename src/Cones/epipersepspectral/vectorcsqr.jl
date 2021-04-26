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

    @inbounds @views for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        r = arr[3:end, j]

        viq = q / v
        @. ξb = ζivi * ∇2h_viw * (-viq * w + r)
        χ = get_χ(p, q, r, cone)
        ζi2χ = ζi2 * χ

        prod[1, j] = ζi2χ
        prod[2, j] = -σ * ζi2χ - dot(viw, ξb) + viq / v
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
    ζi∇h_viw = cache.ζi∇h_viw
    wi = cache.wi
    ζivi = ζi / v
    ζivi2 = ζivi / v

    m = inv.(ζivi * ∇2h_viw + abs2.(wi))
    α = m .* ∇h_viw
    β = dot(∇h_viw, α)
    ζ2β = ζ^2 + β

    w∇2h_viw = ζivi2 * w .* ∇2h_viw
    γ = m .* w∇2h_viw
    c1 = (σ + dot(∇h_viw, γ)) / ζ2β

    c5 = -inv(ζ2β)
    Yu = c5 * α
    Yv = c1 * α - γ

    c3 = ζi2 * σ
    c4 = ζi2 * β
    Zuu = ζi2 - c4 / ζ2β
    Zvu = -c3 + c1 * c4 - ζi2 * dot(γ, ∇h_viw)
    Zvv = (inv(v) + dot(w, w∇2h_viw)) / v + abs2(ζi * σ) + dot(w∇2h_viw - c3 * ∇h_viw, Yv)

    # Hiuu, Hiuv, Hivv
    DZi = inv(Zuu * Zvv - Zvu^2)
    Hiuu = Hi[1, 1] = DZi * Zvv
    Hiuv = Hi[1, 2] = -DZi * Zvu
    Hivv = Hi[2, 2] = DZi * Zuu

    @inbounds for j in 1:d
        Yu_j = Yu[j]
        Yv_j = Yv[j]
        j2 = 2 + j

        # Hiuw
        Hi1j = Hi[1, j2] = -Hiuu * Yu_j - Hiuv * Yv_j

        # Hivw
        Hi2j = Hi[2, j2] = -Hiuv * Yu_j - Hivv * Yv_j

        # Hiww
        for i in 1:j
            Hi[2 + i, j2] = Yu_j * α[i] - Yu[i] * Hi1j - Yv[i] * Hi2j
        end
        Hi[j2, j2] += m[j]
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSepSpectral{VectorCSqr{T}}) where T
    # cone.hess_aux_updated || update_hess_aux(cone) # TODO
    hess(cone) # TODO
    @assert cone.hess_updated
    v = cone.point[2]
    cache = cone.cache



    Hi = inv_hess(cone)
    mul!(prod, Hi, arr)

    # # TODO @inbounds
    # for j in 1:size(arr, 2)
    #
    #
    # end

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
    ∇3hξξ = zeros(d)

    p = dir[1]
    q = dir[2]
    @views r = dir[3:end]

    viq = q / v
    @. ξ = -viq * w + r
    ζivi = ζi / v
    @. ξb = ζivi * ∇2h_viw * ξ
    χ = get_χ(p, q, r, cone)
    ζiχ = ζi * χ
    ζiχpviq = ζiχ + viq

    ξbξ = dot(ξb, ξ) / 2
    ξbviw = dot(ξb, viw)
    c1 = ζi * (ζiχ^2 + ξbξ)

    ζivi2 = ζi / v / v / 2
    @. ∇3hξξ = ζivi2 * ∇3h_viw .* ξ .* ξ

    corr[1] = c1
    corr[2] = -c1 * σ - ζiχpviq * ξbviw + (ξbξ + viq^2) / v + dot(∇3hξξ, viw)
    @. corr[3:end] = -c1 * ∇h_viw + ζiχpviq * ξb - ∇3hξξ + abs2(r * wi) * wi

    return corr
end

function get_χ(
    p::T,
    q::T,
    r::AbstractVector{T},
    cone::EpiPerSepSpectral{VectorCSqr{T}},
    ) where {T <: Real}
    cache = cone.cache
    return p - cache.σ * q - dot(cache.∇h_viw, r)
end
