#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier from Constructing self-concordant barriers for convex cones by Yu. Nesterov
-log(prod_i(w_i^alpha_i) - u) - sum_i(log(w_i))
=#

mutable struct HypoGeomean2{T <: Real} <: Cone{T}
    use_3order_corr::Bool
    use_dual::Bool
    dim::Int
    alpha::Vector{T}
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    wprod::T
    wprodu::T
    tmpn::Vector{T}
    correction::Vector{T}

    function HypoGeomean2{T}(
        alpha::Vector{T},
        is_dual::Bool;
        use_3order_corr::Bool = true,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        dim = length(alpha) + 1
        @assert dim >= 2
        @assert all(ai > 0 for ai in alpha)
        tol = 1e3 * eps(T)
        @assert sum(alpha) ≈ 1 atol=tol rtol=tol
        cone = new{T}()
        cone.use_3order_corr = use_3order_corr
        cone.use_dual = is_dual
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoGeomean2{T}(alpha::Vector{T}) where {T <: Real} = HypoGeomean2{T}(alpha, false)

use_3order_corr(cone::HypoGeomean2) = cone.use_3order_corr

function setup_data(cone::HypoGeomean2{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.tmpn = zeros(T, dim - 1)
    cone.correction = zeros(T, dim)
    return
end

get_nu(cone::HypoGeomean2) = cone.dim

# TODO work out how to get central ray
function set_initial_point(arr::AbstractVector{T}, cone::HypoGeomean2{T}) where {T}
    (arr[1], w) = get_central_ray_hypogeomean2(cone.alpha)
    arr[2:end] .= w
    return arr
end

function update_feas(cone::HypoGeomean2)
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    if all(wi -> wi > 0, w)
        # wprod always calculated because used in update_grad
        cone.wprod = sum(cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha))
        cone.is_feas = (u <= 0) || (cone.wprod > log(u))
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoGeomean2)
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    wprod = cone.wprod = exp(cone.wprod)
    wprodu = cone.wprodu = cone.wprod - u
    cone.grad[1] = inv(wprodu)
    @. cone.tmpn = wprod * cone.alpha / w / wprodu
    @. cone.grad[2:end] = -cone.tmpn - 1 / w
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoGeomean2)
    @assert cone.grad_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    alpha = cone.alpha
    wprod = cone.wprod
    wprodu = cone.wprodu
    wwprodu = wprod / wprodu
    tmpn = cone.tmpn
    H = cone.hess.data

    H[1, 1] = cone.grad[1] / wprodu
    @inbounds for j in eachindex(w)
        j1 = j + 1
        fact = tmpn[j]
        H[1, j1] = -fact / wprodu
        @inbounds for i in 1:(j - 1)
            H[i + 1, j1] = fact * alpha[i] / w[i] * (wwprodu - 1)
        end
        H[j1, j1] = fact * (1 - alpha[j] + alpha[j] * wwprodu) / w[j] + inv(w[j]) / w[j]
    end
    cone.hess_updated = true
    return cone.hess
end

# TODO make more efficient in math and code
function correction(cone::HypoGeomean2, s_sol::AbstractVector, z_sol::AbstractVector)
    @show "here"
    if !cone.hess_updated
        update_hess(cone)
    end
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    wprod = cone.wprod
    wprodu = cone.wprodu
    wwprodu = wprod / wprodu
    alpha = cone.alpha
    Hinv_z = inv_hess(cone) * z_sol
    corr = cone.correction
    corr .= 0
    for i in eachindex(w), j in eachindex(w), k in eachindex(w)
        @show (i, j, k)
        if i == j == k == 1
            corr[k + 1] += wprodu * s_sol[i] * Hinv_z[j]
        elseif i == j == 1
            corr[k + 1] += alpha[k] * wprod / w[k] / wprodu ^ 3 * s_sol[i] * Hinv_z[j]
        elseif i == k == 1
            corr[k + 1] += alpha[j] * wprod / w[j] / wprodu ^ 3 * s_sol[i] * Hinv_z[j]
        elseif j == k == 1
            corr[k + 1] += alpha[i] * wprod / w[i] / wprodu ^ 3 * s_sol[i] * Hinv_z[j]
        elseif i == 1 && j == k
            corr[k + 1] += wprod * alpha[j] / abs2(wprodu) / abs2(w[j]) * (-alpha[j] + 1 + wwprodu * alpha[j]) * s_sol[i] * Hinv_z[j]
        elseif (j == 1 && i == k) || ( k == 1 && i == j)
            corr[k + 1] += wprod * alpha[i] / abs2(wprodu) / abs2(w[i]) * (-alpha[i] + 1 + wwprodu * alpha[i]) * s_sol[i] * Hinv_z[j]
        elseif i == 1
            corr[k + 1] += wwprodu / wprodu * alpha[j] * alpha[k] / w[j] / w[k] * (wwprodu - 1) * s_sol[i] * Hinv_z[j]
        elseif j == 1
            corr[k + 1] += wwprodu / wprodu * alpha[i] * alpha[k] / w[i] / w[k] * (wwprodu - 1) * s_sol[i] * Hinv_z[j]
        elseif k == 1
            corr[k + 1] += wwprodu / wprodu * alpha[j] * alpha[i] / w[j] / w[i] * (wwprodu - 1) * s_sol[i] * Hinv_z[j]
        elseif i == j == k
            corr[k + 1] += (wwprodu * alpha[i] / (w[i]) ^ 3 * (wwprodu * alpha[i] * (alpha[i] - 1) -
                (alpha[i] - 1) * (alpha[i] - 2) - abs2(wwprodu) * abs2(alpha[i]) + 2 * alpha[i] * (alpha[i] - 1) * wwprodu) - 2 / w[i] ^ 3) * s_sol[i] * Hinv_z[j]
        elseif i == j
            corr[k] += (wwprodu * alpha[i] * alpha[k] / w[i] / w[k] * ((alpha[i] - 1) / abs2(w[i]) * (wwprodu - 1) + wwprodu * (2 - prod / w[i]))) * s_sol[i] * Hinv_z[j]
        elseif i == k
            corr[k + 1] += (wwprodu * alpha[i] * alpha[j] / w[i] / w[j] * ((alpha[i] - 1) / abs2(w[i]) * (wwprodu - 1) + wwprodu * (2 - prod / w[i]))) * s_sol[i] * Hinv_z[j]
        elseif j == k
            corr[k + 1] += (wwprodu * alpha[i] * alpha[j] / w[i] / w[j] * ((alpha[j] - 1) / abs2(w[j]) * (wwprodu - 1) + wwprodu * (2 - prod / w[j]))) * s_sol[i] * Hinv_z[j]
        else
            corr[k + 1] += (alpha[i] * alpha[j] * alpha[k] / w[i] / w[j] / w[k] * wwprodu * (3 * wwprodu - 1 - abs2(wwprodu))) * s_sol[i] * Hinv_z[j]
        end
    end

    return corr
end

# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_hypogeomean2(alpha::Vector{<:Real})
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
