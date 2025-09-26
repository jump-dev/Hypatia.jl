#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

"""
$(TYPEDEF)

Hypograph of weighted power mean cone parametrized by powers `α` in the unit
simplex.

    $(FUNCTIONNAME){T}(α::Vector{T}, use_dual::Bool = false)
"""
mutable struct HypoPowerMean{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    α::Vector{T}

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

    ϕ::T
    ζ::T
    tempw1::Vector{T}
    tempw2::Vector{T}

    function HypoPowerMean{T}(α::Vector{T}; use_dual::Bool = false) where {T <: Real}
        dim = length(α) + 1
        @assert dim >= 2
        @assert all(ai > 0 for ai in α)
        @assert sum(α) ≈ 1
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.α = α
        return cone
    end
end

function setup_extra_data!(cone::HypoPowerMean{T}) where {T <: Real}
    cone.tempw1 = zeros(T, cone.dim - 1)
    cone.tempw2 = zero(cone.tempw1)
    return cone
end

get_nu(cone::HypoPowerMean) = cone.dim

function set_initial_point!(
    arr::AbstractVector{T},
    cone::HypoPowerMean{T},
) where {T <: Real}
    # get closed form central ray if all powers are equal, else use fitting
    d = cone.dim - 1
    if all(isequal(inv(T(d))), cone.α)
        (u, w) = get_central_ray_hypogeomean(T, d)
    else
        (u, w) = get_central_ray_hypopowermean(cone.α)
    end
    arr[1] = u
    arr[2:end] .= w
    return arr
end

function update_feas(cone::HypoPowerMean{T}) where {T <: Real}
    @assert !cone.feas_updated
    u = cone.point[1]
    @views w = cone.point[2:end]

    if all(>(eps(T)), w)
        cone.ϕ = exp(sum(α_i * log(w_i) for (α_i, w_i) in zip(cone.α, w)))
        cone.ζ = cone.ϕ - u
        cone.is_feas = (cone.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoPowerMean{T}) where {T <: Real}
    u = cone.dual_point[1]
    @views w = cone.dual_point[2:end]
    α = cone.α

    if u < -eps(T) && all(>(eps(T)), w)
        sumlog = sum(α_i * log(w_i / α_i) for (α_i, w_i) in zip(α, w))
        return (exp(sumlog) + u > eps(T))
    end

    return false
end

function update_grad(cone::HypoPowerMean)
    @assert cone.is_feas
    @views w = cone.point[2:end]
    g = cone.grad
    ζ = cone.ζ

    g[1] = inv(ζ)
    ζiϕ = -cone.ϕ / ζ
    @. g[2:end] = (ζiϕ * cone.α - 1) / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPowerMean)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    @views w = cone.point[2:end]
    α = cone.α
    H = cone.hess.data
    ζ = cone.ζ
    αwi = cone.tempw1
    ζiϕ = cone.ϕ / ζ
    ζiϕ1 = ζiϕ - 1
    @. αwi = α ./ w

    H[1, 1] = ζ^-2

    @inbounds for j in eachindex(w)
        j1 = j + 1
        w_j = w[j]
        α_j = α[j]
        c3 = ζiϕ * αwi[j]
        H[1, j1] = -c3 / ζ

        c2 = c3 * ζiϕ1
        for i in 1:(j - 1)
            H[i + 1, j1] = c2 * αwi[i]
        end
        H[j1, j1] = (c3 * (1 + α_j * ζiϕ1) + inv(w_j)) / w_j
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoPowerMean{T},
) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    α = cone.α
    ζ = cone.ζ
    rwi = cone.tempw1
    ζiϕ = cone.ϕ / ζ

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        @. rwi = r / w

        c0 = dot(rwi, α)
        c1 = ζiϕ * c0 - p / ζ
        prod[1, j] = c1 / -ζ
        c2 = c1 - c0
        @. prod[2:end, j] = (α * ζiϕ * (c2 + rwi) + rwi) / w
    end

    return prod
end

function update_inv_hess(cone::HypoPowerMean)
    @assert cone.grad_updated
    @assert !cone.inv_hess_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    u = cone.point[1]
    @views w = cone.point[2:end]
    Hi = cone.inv_hess.data
    ϕ = cone.ϕ
    ζiϕ = ϕ / cone.ζ
    tempw1 = cone.tempw1
    tempw2 = cone.tempw2
    α = cone.α

    @. tempw1 = 1 + α * ζiϕ
    s1 = sum(abs2(α_i) / t_i for (α_i, t_i) in zip(α, tempw1))
    s2 = 1 - ζiϕ * s1
    ϕs2i = ϕ / s2
    Hi11 = (ϕ - u)^2 + s1 * ϕ * ϕs2i
    @. tempw2 = w / tempw1
    @. tempw1 = α * tempw2

    @views mul!(Hi[2:end, 2:end], tempw1, tempw1', ζiϕ / s2, false)
    @. Hi[1, 2:end] = ϕs2i * tempw1
    Hi[1, 1] = (ϕ - u)^2 + s1 * ϕ * ϕs2i
    @inbounds for j in eachindex(w)
        Hi[j + 1, j + 1] += w[j] * tempw2[j]
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoPowerMean{T},
) where {T <: Real}
    @assert cone.grad_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    ϕ = cone.ϕ
    ζiϕ = ϕ / cone.ζ
    tempw1 = cone.tempw1
    tempw2 = cone.tempw2
    α = cone.α

    @. tempw1 = 1 + α * ζiϕ
    s1 = sum(abs2(α_i) / t_i for (α_i, t_i) in zip(α, tempw1))
    s2 = 1 - ζiϕ * s1
    ϕs2i = ϕ / s2
    Hi11 = (ϕ - u)^2 + s1 * ϕ * ϕs2i
    @. tempw2 = w / tempw1
    @. tempw1 = α * tempw2
    @. tempw2 *= w

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        s3 = dot(r, tempw1)
        prod[1, j] = Hi11 * p + ϕs2i * s3
        cj = ϕs2i * (p + s3 / cone.ζ)
        @. prod[2:end, j] = cj * tempw1 + r * tempw2
    end

    return prod
end

function dder3(cone::HypoPowerMean{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    dder3 = cone.dder3
    p = dir[1]
    @views r = dir[2:end]
    α = cone.α
    ϕ = cone.ϕ
    ζ = cone.ζ
    rwi = cone.tempw1
    ζiϕ = ϕ / ζ

    @. rwi = r / w
    c0 = dot(rwi, α)
    c6 = sum(abs2(rwi_i) * α_i for (rwi_i, α_i) in zip(rwi, α))
    ζiχ = (p - ϕ * c0) / ζ
    c1 = ζiχ^2 + ζiϕ * (c6 - abs2(c0)) / 2
    c7 = ζiϕ * (c1 - c6 / 2 + c0 * (ζiχ + c0 / 2))
    c8 = -ζiϕ * (ζiχ + c0)

    dder3[1] = -c1 / ζ
    @. dder3[2:end] = (α * (c7 + rwi * (c8 + ζiϕ * rwi)) + abs2(rwi)) / w

    return dder3
end

function get_central_ray_hypopowermean(α::Vector{T}) where {T <: AbstractFloat}
    if length(α) == 1
        s = 1 / √T(2)
    else
        s = (√T(5) - 1) / 2
        tol = sqrt(eps(T))
        maxiter = 2ceil(log2(-log2(tol)))
        counter = 0
        while counter < maxiter
            counter += 1
            step = _newton_ratio_powermean(s, α)
            s -= step
            if abs(step) < tol
                break
            end
        end
        counter == maxiter && error("Failed to compute initial point.")
    end
    u = -√(1 - s)
    w = sqrt.(s .* α .+ 1)
    return u, w
end

function _newton_ratio_powermean(s, α)
    logf = 2log(s) - log(1 - s) - sum(αi * log(αi * s + 1) for αi in α)
    dlogf = 2 / s + 1 / (1 - s) - sum(αi^2 / (αi * s + 1) for αi in α)
    return logf / dlogf
end
