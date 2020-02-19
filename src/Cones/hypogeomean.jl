#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier from "Constructing self-concordant barriers for convex cones" by Yu. Nesterov
-log(prod_i(w_i^alpha_i) - u) - sum_i(log(w_i))
=#

mutable struct HypoGeomean{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    alpha::Vector{T}
    point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    wprod::T
    wprodu::T

    function HypoGeomean{T}(
        alpha::Vector{T},
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        dim = length(alpha) + 1
        @assert dim >= 2
        @assert all(ai > 0 for ai in alpha)
        @assert sum(alpha) ≈ 1
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoGeomean{T}(alpha::Vector{T}) where {T <: Real} = HypoGeomean{T}(alpha, false)

function setup_data(cone::HypoGeomean{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    return
end

get_nu(cone::HypoGeomean) = cone.dim

function set_initial_point(arr::AbstractVector{T}, cone::HypoGeomean{T}) where {T}
    # get closed form central ray if all powers are equal, else use fitting
    if all(isequal(inv(T(cone.dim - 1))), cone.alpha)
        n = cone.dim - 1
        c = sqrt(T(5 * n ^ 2 + 2 * n + 1))
        arr[1] = -sqrt((-c + 3 * n + 1) / T(2 * n + 2))
        arr[2:end] .= (c - n + 1) / sqrt(T(n + 1) * (-2 * c + 6 * n + 2))
    else
        (arr[1], w) = get_central_ray_hypogeomean(cone.alpha)
        arr[2:end] .= w
    end
    return arr
end

function update_feas(cone::HypoGeomean)
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)

    if all(>(zero(u)), w)
        cone.wprod = exp(sum(cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha)))
        cone.wprodu = cone.wprod - u
        cone.is_feas = (u < 0) || (wprodu > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoGeomean)
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)

    cone.grad[1] = inv(cone.wprodu)
    wwprodu = -cone.wprod / cone.wprodu
    @. cone.grad[2:end] = (wwprodu * cone.alpha - 1) / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoGeomean)
    @assert cone.grad_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    alpha = cone.alpha
    wprodu = cone.wprodu
    wwprodu = cone.wprod / wprodu
    wwprodum1 = wwprodu - 1
    H = cone.hess.data

    H[1, 1] = cone.grad[1] / wprodu
    @inbounds for j in eachindex(w)
        j1 = j + 1
        wj = w[j]
        aj = alpha[j]
        awwwprodu = wwprodu / wj * aj
        H[1, j1] = -awwwprodu / wprodu
        awwwprodum1 = awwwprodu * wwprodum1
        @inbounds for i in 1:(j - 1)
            H[i + 1, j1] = awwwprodum1 * alpha[i] / w[i]
        end
        H[j1, j1] = (awwwprodu * (1 + aj * wwprodum1) + inv(wj)) / wj
    end

    cone.hess_updated = true
    return cone.hess
end

# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_hypogeomean(alpha::Vector{<:Real})
    wdim = length(alpha)
    # predict each w_i given alpha_i and n
    w = zeros(wdim)
    if wdim == 1
        w .= 1.306563
    elseif wdim == 2
        @. w = 1.005320 + 0.298108 * alpha
    elseif wdim <= 5
        @. w = 1.0049327 - 0.0006020 * wdim + 0.2998672 * alpha
    elseif wdim <= 20
        @. w = 1.001146 - 4.463046e-05 * wdim + 3.034014e-01 * alpha
    elseif wdim <= 100
        @. w = 1.000066 - 5.202270e-07 * wdim + 3.074873e-01 * alpha
    else
        @. w = 1 + 3.086695e-01 * alpha
    end
    # get u in closed form from w
    p = exp(sum(alpha[i] * log(w[i]) for i in eachindex(alpha)))
    u = sum(p .- alpha .* p ./ (abs2.(w) .- 1)) / wdim
    return [u, w]
end
