#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier from "Constructing self-concordant barriers for convex cones" by Yu. Nesterov
-log(prod_i(w_i^alpha_i) - u) - sum_i(log(w_i))
=#

mutable struct HypoGeomean{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    alpha::Vector{T}
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    dual_grad_updated::Bool
    dual_grad_inacc::Bool
    hess_updated::Bool
    scal_hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    dual_grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    scal_hess
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    wprod::T
    z::T
    dual_wprod::T
    dual_z::T

    function HypoGeomean{T}(
        alpha::Vector{T};
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        dim = length(alpha) + 1
        @assert dim >= 2
        @assert all(ai > 0 for ai in alpha)
        @assert sum(alpha) ≈ 1
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

# reset_data(cone::HypoGeomean) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = cone.scal_hess_updated = false)

function setup_data(cone::HypoGeomean{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.dual_grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.scal_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    return
end

get_nu(cone::HypoGeomean) = cone.dim

use_correction(cone::HypoGeomean) = true

function set_initial_point(arr::AbstractVector{T}, cone::HypoGeomean{T}) where {T}
    # get closed form central ray if all powers are equal, else use fitting
    if all(isequal(inv(T(cone.dim - 1))), cone.alpha)
        n = cone.dim - 1
        c = sqrt(T(5 * n ^ 2 + 2 * n + 1))
        arr[1] = -sqrt((-c + 3 * n + 1) / T(2 * n + 2))
        arr[2:end] .= (c - n + 1) / sqrt(T(n + 1) * (-2 * c + 6 * n + 2))
    else
        (arr[1], w) = get_central_ray_hypogeomean(cone.alpha)
        arr[2:end] = w
    end
    return arr
end

function update_feas(cone::HypoGeomean{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)

    if all(>(eps(T)), w)
        cone.wprod = exp(sum(cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha)))
        cone.z = cone.wprod - u
        cone.is_feas = (cone.z > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_dual_feas(cone::HypoGeomean{T}) where {T}
    u = cone.dual_point[1]
    w = view(cone.dual_point, 2:cone.dim)
    alpha = cone.alpha
    if u < -eps(T) && all(>(eps(T)), w)
        cone.dual_wprod = exp(sum(alpha[i] * log(w[i] / alpha[i]) for i in eachindex(alpha)))
        cone.dual_z = cone.dual_wprod + u
        return (cone.dual_z > eps(T))
    end
    return false
end

function update_grad(cone::HypoGeomean)
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)

    cone.grad[1] = inv(cone.z)
    wwprodu = -cone.wprod / cone.z
    @. cone.grad[2:end] = (wwprodu * cone.alpha - 1) / w

    cone.grad_updated = true
    return cone.grad
end

# function update_dual_grad(cone::HypoGeomean{T}, ::T) where {T <: Real}
#     #  TODO assert dual_feas_updated
#     u = cone.dual_point[1]
#     w = view(cone.dual_point, 2:cone.dim)
#     cone.dual_grad_inacc = false
#
#     cone.dual_grad[1] = -inv(cone.dual_z) + inv(u)
#     @. cone.dual_grad[2:end] = -cone.dual_wprod / cone.z / w * cone.alpha - inv(w) * (1 - cone.alpha)
#     @show dot(cone.dual_point, cone.dual_grad)
#
#     cone.dual_grad_updated = true
#     @show cone.dual_grad
#     return cone.dual_grad
# end

function update_hess(cone::HypoGeomean)
    @assert cone.grad_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    alpha = cone.alpha
    z = cone.z
    wwprodu = cone.wprod / z
    wwprodum1 = wwprodu - 1
    H = cone.hess.data

    H[1, 1] = cone.grad[1] / z
    @inbounds for j in eachindex(w)
        j1 = j + 1
        wj = w[j]
        aj = alpha[j]
        awwwprodu = wwprodu / wj * aj
        H[1, j1] = -awwwprodu / z
        awwwprodum1 = awwwprodu * wwprodum1
        @inbounds for i in 1:(j - 1)
            H[i + 1, j1] = awwwprodum1 * alpha[i] / w[i]
        end
        H[j1, j1] = (awwwprodu * (1 + aj * wwprodum1) + inv(wj)) / wj
    end

    cone.hess_updated = true
    return cone.hess
end

# # TODO move into new cone with equal alphas
# function update_inv_hess(cone::HypoGeomean{T}) where {T}
#     @assert all(isequal(inv(T(cone.dim - 1))), cone.alpha)
#     @assert !cone.inv_hess_updated
#     u = cone.point[1]
#     w = view(cone.point, 2:cone.dim)
#     n = cone.dim - 1
#     wprod = cone.wprod
#     H = cone.inv_hess.data
#     denom = n * (n + 1) * wprod - abs2(n) * u
#     H[1, 1] = (n + 1) * abs2(wprod) / n + abs2(u) - 2 * wprod * u
#     H[1, 2:end] = wprod .* w / n
#     H[2:end, 2:end] = wprod * w * w'
#     H[2:end, 2:end] .+= Diagonal(abs2.(w) .* cone.z .* abs2(n))
#     H[2:end, 2:end] /= denom
#
#     cone.inv_hess_updated = true
#     return cone.inv_hess
# end
#
# function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::HypoGeomean{T}) where {T}
#     @assert all(isequal(inv(T(cone.dim - 1))), cone.alpha)
#     dim = cone.dim
#     u = cone.point[1]
#     w = view(cone.point, 2:dim)
#     n = dim - 1
#     wprod = cone.wprod
#
#     @inbounds for j in 1:size(prod, 2)
#         @views pa = dot(w, arr[2:dim, j]) * wprod
#         @. prod[2:dim, j] = pa * w
#         prod[1, j] = pa / n
#     end
#     @. @views prod[1, :] += ((n + 1) * abs2(wprod) / n + abs2(u) - 2 * wprod * u) * arr[1, :]
#     @. @views prod[2:dim, :] += (abs2.(w) .* cone.z .* abs2(n)) * arr[2:dim, :]
#     @. @views prod[2:dim, :] /= (n * (n + 1) * wprod - abs2(n) * u)
#     @views mul!(prod[2:dim, :], w, arr[1, :]', wprod / n, true)
#     return prod
# end

function correction2(cone::HypoGeomean{T}, primal_dir::AbstractVector{T}) where {T <: Real}
    dim = cone.dim
    u = cone.point[1]
    w = view(cone.point, 2:dim)
    pi = cone.wprod # TODO rename
    z = cone.z
    alpha = cone.alpha
    corr = cone.correction
    u_dir = primal_dir[1]
    w_dir = view(primal_dir, 2:dim)

    piz = pi / z
    wdw = w_dir ./ w
    udz = u_dir / z
    uuw1 = -2 * udz * piz
    awdw = dot(alpha, wdw)
    uww1 = awdw * piz * (2 * piz - 1)
    awdw2 = sum(alpha[i] * abs2(wdw[i]) for i in eachindex(alpha))
    corr[1] = (abs2(udz) + uuw1 * awdw + (uww1 * awdw + piz * awdw2) / 2) / -z
    www1 = piz * (1 - piz)
    all1 = (uuw1 * u_dir / z + www1 * awdw2 + awdw * u * piz * (1 - 2 * piz) / z * awdw) / -2 - udz * uww1
    all2 = www1 * awdw + udz * piz
    all3 = www1 + piz * u / z
    @views wcorr = corr[2:end]
    @. wcorr = all1 * alpha
    @. wcorr += wdw * (((all3 * alpha + piz) * alpha + 1) * wdw - all2 * alpha) # TODO check this is fast - if not, use an explicit loop
    wcorr ./= w

    return corr
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
