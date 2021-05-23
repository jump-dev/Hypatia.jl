"""
$(TYPEDEF)

Hypograph of weighted power mean cone parametrized by powers `alpha` in the unit
simplex.

    $(FUNCTIONNAME){T}(alpha::Vector{T}, use_dual::Bool = false)
"""
mutable struct HypoPowerMean{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    alpha::Vector{T}

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

    ϕ::T
    ζ::T
    ζi::T
    tempw1::Vector{T}
    tempw2::Vector{T}

    function HypoPowerMean{T}(
        alpha::Vector{T};
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        dim = length(alpha) + 1
        @assert dim >= 2
        @assert all(ai > 0 for ai in alpha)
        @assert sum(alpha) ≈ 1
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data!(cone::HypoPowerMean{T}) where {T <: Real}
    cone.tempw1 = zeros(T, cone.dim - 1)
    cone.tempw2 = zeros(T, cone.dim - 1)
    return cone
end

get_nu(cone::HypoPowerMean) = cone.dim

function set_initial_point!(arr::AbstractVector{T}, cone::HypoPowerMean{T}) where T
    # get closed form central ray if all powers are equal, else use fitting
    if all(isequal(inv(T(cone.dim - 1))), cone.alpha)
        n = cone.dim - 1
        c = sqrt(T(5 * n ^ 2 + 2 * n + 1))
        arr[1] = -sqrt((-c + 3 * n + 1) / T(2 * n + 2))
        @views arr[2:end] .= (c - n + 1) / sqrt(T(n + 1) * (-2 * c + 6 * n + 2))
    else
        (arr[1], w) = get_central_ray_hypopowermean(cone.alpha)
        @views arr[2:end] = w
    end
    return arr
end

function update_feas(cone::HypoPowerMean{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    alpha = cone.alpha

    if all(>(eps(T)), w)
        @inbounds cone.ϕ = exp(sum(alpha[i] * log(w[i])
            for i in eachindex(alpha)))
        cone.ζ = cone.ϕ - u
        cone.is_feas = (cone.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoPowerMean{T}) where T
    u = cone.dual_point[1]
    @views w = cone.dual_point[2:end]
    alpha = cone.alpha

    @inbounds if u < -eps(T) && all(>(eps(T)), w)
        return (exp(sum(alpha[i] * log(w[i] / alpha[i])
            for i in eachindex(alpha))) + u > eps(T))
    end

    return false
end

function update_grad(cone::HypoPowerMean)
    @assert cone.is_feas
    u = cone.point[1]
    @views w = cone.point[2:end]

    ζi = cone.ζi = inv(cone.ζ)
    cone.grad[1] = ζi
    wϕu = -cone.ϕ * ζi
    @. @views cone.grad[2:end] = (wϕu * cone.alpha - 1) / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPowerMean)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    u = cone.point[1]
    @views w = cone.point[2:end]
    alpha = cone.alpha
    ζ = cone.ζ
    ζi = cone.ζi
    aw = alpha ./ w # TODO
    wϕu = cone.ϕ / ζ
    wϕum1 = wϕu - 1

    H[1, 1] = abs2(ζi)
    @inbounds for j in eachindex(w)
        j1 = j + 1
        wj = w[j]
        aj = alpha[j]
        awwϕu = wϕu * aw[j]
        H[1, j1] = -awwϕu / ζ
        awwϕum1 = awwϕu * wϕum1
        @inbounds for i in 1:(j - 1)
            H[i + 1, j1] = awwϕum1 * aw[i]
        end
        H[j1, j1] = (awwϕu * (1 + aj * wϕum1) + inv(wj)) / wj
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPowerMean,
    )
    @assert cone.grad_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    alpha = cone.alpha
    ζi = cone.ζi
    ϕ = cone.ϕ
    rwi = cone.tempw1

    @inbounds @views for j in 1:size(arr, 2)
        p = arr[1, j]
        r = arr[2:end, j]
        @. rwi = r / w
        c0 = dot(rwi, alpha)
        c1 = -ζi * (p - ϕ * c0) * ζi
        prod[1, j] = -c1
        # ∇2h[r] = ϕ * (c0 - rwi) / w * alpha
        @. prod[2:end, j] = ϕ * (c1 - ζi * (c0 - rwi)) * alpha / w + r / w / w
    end

    return prod
end

function dder3(cone::HypoPowerMean, dir::AbstractVector)
    @assert cone.grad_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    dder3 = cone.dder3
    alpha = cone.alpha
    p = dir[1]
    @views r = dir[2:end]
    ϕ = cone.ϕ
    ζi = cone.ζi

    rwi = cone.tempw1
    @. rwi = r / w
    c0 = dot(rwi, alpha)
    rwi_sqr = sum(abs2(rwij) * aj for (rwij, aj) in zip(rwi, alpha))

    ζiχ = ζi * (p - ϕ * c0)
    ξbξ = ζi * ϕ * (c0^2 - rwi_sqr) / 2
    c1 = -ζi * (ζiχ^2 - ξbξ)

    c2 = -ζi / 2
    w_aux = cone.tempw2
    # ∇2h[r] = ϕ * (c0 - rwi) / w * alpha
    @. w_aux = ζi * ϕ * (c0 - rwi) / w * alpha
    w_aux .*= ζiχ
    # add c2 * ∇3h[r, r]
    @. w_aux -= c2 * ϕ * ((c0 - rwi)^2 - rwi_sqr + rwi^2) * alpha / w

    dder3[1] = c1
    @. dder3[2:end] = (abs2(rwi) - c1 * ϕ * alpha) / w + w_aux

    return dder3
end

# see analysis in
# https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_hypopowermean(alpha::Vector{<:Real})
    w_dim = length(alpha)
    # predict each w_i given alpha_i and n
    w = zeros(w_dim)
    if w_dim == 1
        w .= 1.306563
    elseif w_dim == 2
        @. w = 1.0049885 + 0.2986276 * alpha
    elseif w_dim <= 5
        @. w = 1.0040142949 - 0.0004885108 * w_dim + 0.3016645951 * alpha
    elseif w_dim <= 20
        @. w = 1.001168 - 4.547017e-05 * w_dim + 3.032880e-01 * alpha
    elseif w_dim <= 100
        @. w = 1.000069 - 5.469926e-07 * w_dim + 3.074084e-01 * alpha
    else
        @. w = 1 + 3.086535e-01 * alpha
    end
    # get u in closed form from w
    p = exp(sum(alpha[i] * log(w[i]) for i in eachindex(alpha)))
    u = sum(p .- alpha .* p ./ (abs2.(w) .- 1)) / w_dim
    return [u, w]
end
