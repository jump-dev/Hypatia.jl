#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(u^2 - norm_2(w)^2)

TODO
- try to derive faster neighborhood calculations for this cone specifically
=#

mutable struct EpiNormEucl{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    max_neighborhood::T
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    dist::T

    function EpiNormEucl{T}(
        dim::Int;
        use_dual::Bool = false, # TODO self-dual so maybe remove this option/field?
        max_neighborhood::Real = default_max_neighborhood(),
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        return cone
    end
end

use_heuristic_neighborhood(cone::EpiNormEucl) = false

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    return
end

get_nu(cone::EpiNormEucl) = 2

function set_initial_point(arr::AbstractVector, cone::EpiNormEucl{T}) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

# TODO refac with dual feas check
function update_feas(cone::EpiNormEucl{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views w = cone.point[2:end]
        cone.dist = (abs2(u) - sum(abs2, w)) / 2
        cone.is_feas = (cone.dist > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_dual_feas(cone::EpiNormEucl{T}) where {T}
    u = cone.dual_point[1]
    if u > eps(T)
        w = view(cone.dual_point, 2:cone.dim)
        @views dual_dist = abs2(u) - sum(abs2, cone.dual_point[2:end])
        return (dual_dist > 2 * eps(T))
    end
    return false
end

function update_grad(cone::EpiNormEucl)
    @assert cone.is_feas

    @. cone.grad = cone.point / cone.dist
    cone.grad[1] *= -1

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated

    mul!(cone.hess.data, cone.grad, cone.grad')
    inv_dist = inv(cone.dist)
    @inbounds for j in eachindex(cone.grad)
        cone.hess[j, j] += inv_dist
    end
    cone.hess[1, 1] -= inv_dist + inv_dist

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiNormEucl)
    @assert cone.is_feas

    mul!(cone.inv_hess.data, cone.point, cone.point')
    @inbounds for j in eachindex(cone.grad)
        cone.inv_hess[j, j] += cone.dist
    end
    cone.inv_hess[1, 1] -= cone.dist + cone.dist

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    @inbounds for j in 1:size(prod, 2)
        uj = arr[1, j]
        wj = @view arr[2:end, j]
        ga = (dot(w, wj) - u * uj) / cone.dist
        prod[1, j] = -ga * u - uj
        @. prod[2:end, j] = ga * w + wj
    end
    @. prod ./= cone.dist

    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas

    @inbounds for j in 1:size(prod, 2)
        @views pa = dot(cone.point, arr[:, j])
        @. prod[:, j] = pa * cone.point
    end
    @. @views prod[1, :] -= cone.dist * arr[1, :]
    @. @views prod[2:end, :] += cone.dist * arr[2:end, :]

    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormEucl{T}) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    rt2 = sqrt(T(2))
    distrt2 = cone.dist * rt2
    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist * rt2
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        @views wj = arr[2:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (u * uj - dotwwj) / distrt2
        wmulj = (dotwwj / urtdist - uj) / distrt2
        @. prod[2:end, j] = w * wmulj + wj / rtdist
    end

    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormEucl{T}) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    rt2 = sqrt(T(2))
    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist * rt2
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        @views wj = arr[2:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (u * uj + dotwwj) / rt2
        wmulj = (dotwwj / urtdist + uj) / rt2
        @. prod[2:end, j] = w * wmulj + wj * rtdist
    end

    return prod
end
