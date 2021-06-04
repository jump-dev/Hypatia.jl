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
    tempw::Vector{T}

    function HypoPowerMean{T}(
        α::Vector{T};
        use_dual::Bool = false,
        ) where {T <: Real}
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
    cone.tempw = zeros(T, cone.dim - 1)
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
    αwi = cone.tempw
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
    rwi = cone.tempw
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

function dder3(cone::HypoPowerMean{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    dder3 = cone.dder3
    p = dir[1]
    @views r = dir[2:end]
    α = cone.α
    ϕ = cone.ϕ
    ζ = cone.ζ
    rwi = cone.tempw
    ζiϕ = ϕ / ζ

    @. rwi = r / w
    c0 = dot(rwi, α)
    c6 = sum(abs2(rwij) * α_i for (rwij, α_i) in zip(rwi, α))
    ζiχ = (p - ϕ * c0) / ζ
    c1 = ζiχ^2 + ζiϕ * (c6 - abs2(c0)) / 2
    c7 = ζiϕ * (c1 - c6 / 2 + c0 * (ζiχ + c0 / 2))
    c8 = -ζiϕ * (ζiχ + c0)

    dder3[1] = -c1 / ζ
    @. dder3[2:end] = (α * (c7 + rwi * (c8 + ζiϕ * rwi)) + abs2(rwi)) / w

    return dder3
end

# see analysis in
# https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_hypopowermean(α::Vector{T}) where {T <: Real}
    d = length(α)
    # predict w given α and d
    w = zeros(T, d)
    if d == 1
        w .= 1.306563
    elseif d == 2
        @. w = 1.0049885 + 0.2986276 * α
    elseif d <= 5
        @. w = 1.0040142949 - 0.0004885108 * d + 0.3016645951 * α
    elseif d <= 20
        @. w = 1.001168 - 4.547017e-05 * d + 3.032880e-01 * α
    elseif d <= 100
        @. w = 1.000069 - 5.469926e-07 * d + 3.074084e-01 * α
    else
        @. w = 1 + 3.086535e-01 * α
    end
    # get u in closed form from w
    p = exp(sum(α_i * log(w_i) for (α_i, w_i) in zip(α, w)))
    u = p - p / d * sum(α_i / (abs2(w_i) - 1) for (α_i, w_i) in zip(α, w))
    return (u, w)
end
