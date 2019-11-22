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
    if !cone.hess_updated
        update_hess(cone)
    end
    dim = cone.dim
    u = cone.point[1]
    w = view(cone.point, 2:dim)
    wprod = cone.wprod
    wprodu = cone.wprodu
    wwprodu = wprod / wprodu
    alpha = cone.alpha
    Hinv_z = inv_hess(cone) * z_sol
    corr = cone.correction
    corr .= 0

    # case where i = j = k = 1
    corr[1] += 2 / wprodu ^ 3 * s_sol[1] * Hinv_z[1]
    for i in 2:dim
        i1 = i - 1
        # case where i = k = 1, and j > 1, and symmetric case where j = k = 1, and i > 1
        sz = s_sol[i] * Hinv_z[1] + s_sol[1] * Hinv_z[i]
        corr[1] += (-2 * alpha[i1] * wprod / w[i1] / wprodu ^ 3) * sz
        # cases where k = 1 and i != j and i > 1 or j > 1
        for j in 2:(i - 1)
            j1 = j - 1
            sz = s_sol[i] * Hinv_z[j] + s_sol[j] * Hinv_z[i]
            corr[1] += wwprodu / wprodu * alpha[j1] * alpha[i1] / w[j1] / w[i1] * (2 * wwprodu - 1) * sz
        end
        # case where i = j
        corr[1] += wprod * alpha[i1] / abs2(wprodu) / abs2(w[i1]) * (-alpha[i1] + 1 + 2 * wwprodu * alpha[i1]) * s_sol[i] * Hinv_z[i]
    end

    for k in 2:dim
        k1 = k - 1
        # i = j = 1, k > 1
        corr[k] += -2 * alpha[k1] * wprod / w[k1] / wprodu ^ 3 * s_sol[1] * Hinv_z[1]
        for i in 2:dim
            i1 = i - 1
            sz = (s_sol[i] * Hinv_z[1] +  s_sol[1] * Hinv_z[i])
            if i == k
                # case where j = 1 and i = k, symmetric case is i = 1 and j = k
                corr[k] += wprod * alpha[i1] / abs2(wprodu) / abs2(w[i1]) * (-alpha[i1] + 1 + 2 * wwprodu * alpha[i1]) * sz
            else
                # case where j = 1 and i != k, symmetric case is i = 1 and j != k
                corr[k] += wwprodu / wprodu * alpha[i1] * alpha[k1] / w[i1] / w[k1] * (2 * wwprodu - 1) * sz
            end
            for j in 2:(i - 1)
                j1 = j - 1
                sz = s_sol[i] * Hinv_z[j] + s_sol[j] * Hinv_z[i]
                if i == k || j == k
                    l = (i == k ? j1 : i1)
                    corr[k] += (wwprodu - 1) * (alpha[l] * alpha[k1] * wwprodu / w[l] / abs2(w[k1]) * (alpha[k1] - 1 - 2 * alpha[k1] * wwprodu)) * sz
                else
                    # i, j, k all different
                    corr[k] += (alpha[i1] * alpha[j1] * alpha[k1] / w[i1] / w[j1] / w[k1] * wwprodu * (3 * wwprodu - 1 - abs2(wwprodu))) * sz
                end
            end
            # case where i = j
            sz = s_sol[i] * Hinv_z[i]
            # case where i == j == k
            if i == k
                corr[k] += (wwprodu * alpha[i1] / (w[i1]) ^ 3 * (wwprodu * alpha[i1] * (alpha[i1] - 1) -
                    (alpha[i1] - 1) * (alpha[i1] - 2) - 2 * abs2(wwprodu) * abs2(alpha[i1]) + 2 * alpha[i1] * (alpha[i1] - 1) * wwprodu) - 2 / w[i1] ^ 3) * sz
            # case where i = j != k
            else
                corr[k] += (wwprodu - 1) * (alpha[k1] * alpha[i1] * wwprodu / w[k1] / abs2(w[i1]) * (alpha[i1] - 1 - 2 * alpha[i1] * wwprodu)) * sz
            end
        end
    end
    corr ./= -2

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
