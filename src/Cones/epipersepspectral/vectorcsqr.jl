#=
TODO

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

function set_initial_point(arr::AbstractVector, cone::EpiPerSepSpectral{<:VectorCSqr, F}) where F
    (arr[1], arr[2], w0) = get_initial_point(F, cone.d)
    @views fill!(arr[3:end], w0)
    return arr
end

function update_feas(cone::EpiPerSepSpectral{VectorCSqr{T}, F, T}) where {T, F}
    @assert !cone.feas_updated
    cache = cone.cache
    v = cone.point[2]

    if (v > eps(T)) && all(>(eps(T)), cone.w_view)
        @. cache.viw = cone.w_view / v
        cache.ϕ = h_val(F, cache.viw)
        cache.ζ = cone.point[1] - v * cache.ϕ
        cone.is_feas = (cache.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiPerSepSpectral{<:VectorCSqr{T}, F, T}) where {T, F}
    u = cone.dual_point[1]
    (u < eps(T)) && return false
    @views w = cone.dual_point[3:end]
    # TODO in-place:
    temp = similar(w)
    @. temp = w / u
    h_conj_dom(F, temp) || return false
    v = cone.dual_point[2]
    return (v - u * h_conj(F, temp) > eps(T))
end

function update_grad(cone::EpiPerSepSpectral{<:VectorCSqr, F}) where F
    @assert !cone.grad_updated && cone.is_feas
    grad = cone.grad
    v = cone.point[2]
    cache = cone.cache
    ζi = cache.ζi = inv(cache.ζ)
    ∇h_viw = cache.∇h_viw
    @. ∇h_viw = h_der1(F, cache.viw)
    cache.σ = cache.ϕ - dot(cache.viw, ∇h_viw)
    @. cache.wi = inv(cone.w_view)
    @. cache.ζi∇h_viw = ζi * ∇h_viw

    grad[1] = -ζi
    grad[2] = -inv(v) + ζi * cache.σ
    @. grad[3:end] = -cache.wi + cache.ζi∇h_viw

    cone.grad_updated = true
    return grad
end

function update_hess(cone::EpiPerSepSpectral{<:VectorCSqr, F}) where F
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
    @. ∇2h_viw = h_der2(F, cache.viw)
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


# TODO uses old math from paper with wrong zeta definition
function update_inv_hess(cone::EpiPerSepSpectral{<:VectorCSqr, F}) where F
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
    # @. ∇2h_viw = h_der2(F, cache.viw)
    ζi∇h_viw = cache.ζi∇h_viw
    wi = cache.wi
    ζivi = ζi / v
    ζiσ = ζi * σ


    # TODO delete
    H = cone.hess

    # new
    m = inv.(ζivi * ∇2h_viw + abs2.(wi))
    α = m .* ∇h_viw
    β = dot(∇h_viw, α)

    γ = m .* ∇2h_viw .* w
    # δ = sum(m .* ∇h_viw .* ∇2h_viw .* viw)
    δ = dot(∇h_viw, γ)

    Yu = -inv(ζ^2 + β) * α
    @assert Yu ≈ Symmetric(H[3:end, 3:end]) \ H[1, 3:end]
    Yv = (ζ^(-2) * σ + ζ^(-3) / v^2 * δ) / (1 + ζ^(-2) * β) * α - v^(-2) / ζ * γ
    # Yv = (σ + ζivi / v * δ) / (ζ^2 + β) * α - ζivi / v * γ
    # @show Yv
    # Yv2 = Symmetric(H[3:end, 3:end]) \ H[2, 3:end]
    # @show Yv2
    # @assert Yv ≈ Yv2

    Zuu = ζi2 - inv(ζ^2 + β) / ζ^2 * dot(α, ∇h_viw)
    # @show Zuu / (ζi2 - dot(H[1,3:end], Yu))

    # TODO refac constants with above and below
    Zvu = -ζi2 * σ + (ζi2 * σ + ζ^(-3) / v^2 * δ) / (1 + ζ^(-2) * β) * dot(α, ∇h_viw) / ζ^2 - v^(-2) * dot(γ, ∇h_viw) / ζ^3
    # @show Zvu / (-ζi2 * σ - dot(H[2, 3:end], Yu))

    # c1 = (ζ^(-2) * σ + ζ^(-3) / v^2 * δ) / (1 + ζ^(-2) * β)
    c1 = (σ + ζivi / v * δ) / (ζ^2 + β)
    c2 = ζivi / v
    c3 = ζi2 * σ
    Hvv = v^-2 + abs2(ζi * σ) + ζivi * sum(viw[i]^2 * ∇2h_viw[i] for i in 1:d)
    @assert Hvv ≈ H[2, 2]
    Zvv = Hvv - dot(c3 * ∇h_viw - c2 * ∇2h_viw .* w, c1 * α - c2 * γ)
    # @show Zvv / (Hvv - dot(H[2, 3:end], Yv))


    # Huu, Huv, Hvv
    # Zi = inv(Zuu * Zvv - Zvu^2) * [Zvv -Zvu; -Zvu Zuu]
    DZi = inv(Zuu * Zvv - Zvu^2)
    Hiuu = Hi[1, 1] = DZi * Zvv
    Hiuv = Hi[1, 2] = -DZi * Zvu
    Hivv = Hi[2, 2] = DZi * Zuu

    # Huw, Hvw
    @. Hi[1, 3:end] = -Hiuu * Yu - Hiuv * Yv
    @. Hi[2, 3:end] = -Hiuv * Yu - Hivv * Yv

    # # Hww
    # # ?? TODO copied
    # Ai = Diagonal(inv.(inv(v) / ζ * vec(diff_mat) + [inv(vals[i]) / vals[j] for i in 1:d for j in 1:d]))
    # Hwwi = Ai - ζ^(-2) * Ai * vec(nablaphi) * vec(nablaphi)' * Ai / (1 + ζ^(-2) * dot(vec(nablaphi), Ai, vec(nablaphi)))
    # Hiww = Hwwi + Y * Zi * Y'
    #
    # @inbounds for j in 1:d
    #     j2 = 2 + j
    #     for i in 1:(j - 1)
    #         Hi[2 + i, j2] =
    #     end
    #     Hi[j2, j2] =
    # end

    cone.inv_hess_updated = true
    return cone.inv_hess
end





function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSepSpectral{<:VectorCSqr, F}) where F
    # cone.hess_aux_updated || update_hess_aux(cone) # TODO
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

# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSepSpectral{<:VectorCSqr, F}) where F
#     # cone.hess_aux_updated || update_hess_aux(cone) # TODO
#     v = cone.point[2]
#     cache = cone.cache
#
#     # TODO @inbounds
#     for j in 1:size(arr, 2)
#
#
#     end
#
#     return prod
# end

function correction(cone::EpiPerSepSpectral{<:VectorCSqr{T}, F}, dir::AbstractVector{T}) where {T, F}
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
    @. ∇3h_viw = h_der3(F, cache.viw)
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
    cone::EpiPerSepSpectral{<:VectorCSqr{T}},
    ) where {T <: Real}
    cache = cone.cache
    return p - cache.σ * q - dot(cache.∇h_viw, r)
end
