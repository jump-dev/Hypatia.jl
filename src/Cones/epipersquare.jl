#=
Copyright 2018, Chris Coey and contributors

epigraph of perspective of (half) square function (AKA rotated second-order cone)
(u in R, v in R_+, w in R^n) : u >= v*1/2*norm_2(w/v)^2
note v*1/2*norm_2(w/v)^2 = 1/2*sum_i(w_i^2)/v

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(2*u*v - norm_2(w)^2)

TODO
- try to derive faster neighborhood calculations for this cone specifically
=#

mutable struct EpiPerSquare{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    max_neighborhood::T
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    dual_feas_updated::Bool
    grad_updated::Bool
    dual_grad_updated::Bool
    dual_grad_inacc::Bool
    hess_updated::Bool
    scal_hess_updated::Bool
    nt_updated::Bool
    inv_hess_updated::Bool
    hess_sqrt_prod_updated::Bool
    inv_hess_sqrt_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    dual_grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    scal_hess
    inv_hess::Symmetric{T, Matrix{T}}
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    dist::T
    dual_dist::T
    rtdist::T
    rt_dist_ratio::T
    rt_rt_dist_ratio::T
    denom::T
    hess_sqrt_vec::Vector{T}
    inv_hess_sqrt_vec::Vector{T}
    nt_point::Vector{T} # actually a normalized nt point
    nt_point_sqrt::Vector{T}
    normalized_point::Vector{T}
    normalized_dual_point::Vector{T}

    function EpiPerSquare{T}(
        dim::Int;
        use_dual::Bool = false, # TODO self-dual so maybe remove this option/field?
        max_neighborhood::Real = default_max_neighborhood(),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        return cone
    end
end

reset_data(cone::EpiPerSquare) = (cone.feas_updated = cone.dual_feas_updated = cone.grad_updated = cone.dual_grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_sqrt_prod_updated = cone.inv_hess_sqrt_prod_updated = cone.nt_updated = false)

use_scaling(::EpiPerSquare) = false # TODO update oracles

use_correction(cone::EpiPerSquare) = true

function setup_data(cone::EpiPerSquare{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.dual_grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.scal_hess = zeros(T, dim, dim)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.correction = zeros(T, dim)
    cone.hess_sqrt_vec = zeros(T, dim)
    cone.inv_hess_sqrt_vec = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.nt_point = zeros(T, dim)
    cone.nt_point_sqrt = zeros(T, dim)
    cone.normalized_point = zeros(T, dim)
    cone.normalized_dual_point = zeros(T, dim)
    return
end

get_nu(cone::EpiPerSquare) = 2

function set_initial_point(arr::AbstractVector, cone::EpiPerSquare)
    arr[1:2] .= 1
    arr[3:end] .= 0
    return arr
end

# TODO refac with dual feas check
function update_feas(cone::EpiPerSquare{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]

    if u > eps(T) && v > eps(T)
        w = view(cone.point, 3:cone.dim)
        cone.dist = u * v - sum(abs2, w) / 2
        cone.is_feas = (cone.dist > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_dual_feas(cone::EpiPerSquare{T}) where {T}
    u = cone.dual_point[1]
    v = cone.dual_point[2]
    cone.dual_feas_updated = true

    if u > eps(T) && v > eps(T)
        w = view(cone.dual_point, 3:cone.dim)
        cone.dual_dist = u * v - sum(abs2, w) / 2
        return cone.dual_dist > eps(T)
    end
    return false
end

function update_grad(cone::EpiPerSquare)
    @assert cone.is_feas

    @. cone.grad = cone.point / cone.dist
    g2 = cone.grad[2]
    cone.grad[2] = -cone.grad[1]
    cone.grad[1] = -g2

    cone.grad_updated = true
    return cone.grad
end

function update_dual_grad(cone::EpiPerSquare{T}, ::T) where {T <: Real}
    @assert cone.dual_feas_updated
    cone.dual_grad_inacc = false

    @. cone.dual_grad = cone.dual_point / cone.dual_dist
    g2 = cone.dual_grad[2]
    cone.dual_grad[2] = -cone.dual_grad[1]
    cone.dual_grad[1] = -g2

    cone.dual_grad_updated = true
    return cone.dual_grad
end

function update_hess(cone::EpiPerSquare)
    @assert cone.grad_updated
    H = cone.hess.data

    mul!(H, cone.grad, cone.grad')
    inv_dist = inv(cone.dist)
    @inbounds for j in 3:cone.dim
        H[j, j] += inv_dist
    end
    H[1, 2] -= inv_dist

    cone.hess_updated = true
    return cone.hess
end

function update_scal_hess(cone::EpiPerSquare{T}, mu::T) where {T}
    cone.nt_updated || update_nt(cone)
    H = cone.hess.data

    # NOTE does the barrier need to be scaled by rt2? the dist of nt_point is 1, but dist of its inverse/grad is 2. asymmetry.
    nt_inv = copy(cone.nt_point) * 2 # TODO allocs
    g2 = nt_inv[2]
    nt_inv[2] = -nt_inv[1]
    nt_inv[1] = -g2
    # @show rotated_jdot(cone.nt_point, cone.nt_point)
    # @show rotated_jdot(nt_inv, nt_inv)
    @show dot(cone.nt_point, nt_inv)

    mul!(H, nt_inv, nt_inv', 2, false)
    @inbounds for j in 3:cone.dim
        H[j, j] += 4
    end
    H[1, 2] -= 4

    cone.hess_updated = true
    return cone.hess
end


function update_inv_hess(cone::EpiPerSquare)
    @assert cone.is_feas

    mul!(cone.inv_hess.data, cone.point, cone.point')
    @inbounds for j in 3:cone.dim
        cone.inv_hess.data[j, j] += cone.dist
    end
    cone.inv_hess.data[1, 2] -= cone.dist

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSquare)
    u = cone.point[1]
    v = cone.point[2]
    w = @view cone.point[3:end]

    @inbounds for j in 1:size(prod, 2)
        uj = arr[1, j]
        vj = arr[2, j]
        wj = @view arr[3:end, j]
        ga = (dot(w, wj) - v * uj - u * vj) / cone.dist
        prod[1, j] = -ga * v - vj
        prod[2, j] = -ga * u - uj
        @. prod[3:end, j] = ga * w + wj
    end
    @. prod /= cone.dist

    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSquare)
    @assert cone.is_feas

    @inbounds for j in 1:size(prod, 2)
        @views pa = dot(cone.point, arr[:, j])
        @. prod[:, j] = pa * cone.point
    end
    @. @views prod[1, :] -= cone.dist * arr[2, :]
    @. @views prod[2, :] -= cone.dist * arr[1, :]
    @. @views prod[3:end, :] += cone.dist * arr[3:end, :]

    return prod
end

function update_hess_sqrt_prod(cone::EpiPerSquare)
    @assert cone.is_feas
    @assert !cone.hess_sqrt_prod_updated

    rtdist = cone.rtdist = sqrt(cone.dist)
    cone.denom = 2 * rtdist + cone.point[1] + cone.point[2]
    vec = cone.hess_sqrt_vec
    @. @views vec[3:end] = cone.point[3:end] / rtdist
    vec[1] = -cone.point[2] / rtdist - 1
    vec[2] = -cone.point[1] / rtdist - 1

    cone.hess_sqrt_prod_updated = true
    return
end

function update_inv_hess_sqrt_prod(cone::EpiPerSquare)
    @assert cone.is_feas
    @assert !cone.inv_hess_sqrt_prod_updated

    rtdist = cone.rtdist = sqrt(cone.dist)
    cone.denom = 2 * rtdist + cone.point[1] + cone.point[2]
    vec = cone.inv_hess_sqrt_vec
    copyto!(vec, cone.point)
    vec[1:2] .+= rtdist

    cone.inv_hess_sqrt_prod_updated = true
    return
end

function hess_sqrt_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare{T}) where {T <: Real}
    if !cone.hess_sqrt_prod_updated
        update_hess_sqrt_prod(cone)
    end
    vec = cone.hess_sqrt_vec
    rtdist = cone.rtdist

    @inbounds for j in 1:size(arr, 2)
        @views dotj = dot(vec, arr[:, j]) / cone.denom
        @. prod[:, j] = dotj * vec
    end
    @. @views prod[1, :] -= arr[2, :] / rtdist
    @. @views prod[2, :] -= arr[1, :] / rtdist
    @. @views prod[3:end, :] += arr[3:end, :] / rtdist

    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare{T}) where {T <: Real}
    if !cone.inv_hess_sqrt_prod_updated
        update_inv_hess_sqrt_prod(cone)
    end
    vec = cone.inv_hess_sqrt_vec
    rtdist = cone.rtdist

    @inbounds for j in 1:size(arr, 2)
        @views dotj = dot(vec, arr[:, j]) / cone.denom
        @. prod[:, j] = dotj * vec
    end
    @. @views prod[1, :] -= arr[2, :] * rtdist
    @. @views prod[2, :] -= arr[1, :] * rtdist
    @. @views prod[3:end, :] += arr[3:end, :] * rtdist

    return prod
end

rotated_jdot(x::AbstractVector, y::AbstractVector) = @views x[1] * y[2] + x[2] * y[1] - dot(x[3:end], y[3:end]) # TODO if only used once, don't make separate function

# TODO allocs
function correction2(cone::EpiPerSquare, primal_dir::AbstractVector)
    @assert cone.grad_updated
    dim = cone.dim
    corr = cone.correction
    point = cone.point

    tmp = hess_prod!(cone.nbhd_tmp, primal_dir, cone)
    tmp2 = cone.nbhd_tmp2
    copyto!(tmp2, primal_dir)
    @views tmp2[3:dim] .*= -1
    (tmp2[1], tmp2[2]) = (tmp2[2], tmp2[1])

    corr .= point * dot(primal_dir, tmp)
    corr[3:dim] .*= -1
    (corr[1], corr[2]) = (corr[2], corr[1])
    corr .+= tmp * rotated_jdot(point, primal_dir)
    corr .-= dot(point, tmp) * tmp2
    corr ./= 2 * cone.dist

    return corr
end

function update_nt(cone::EpiPerSquare)
    @assert cone.feas_updated
    cone.dual_feas_updated || update_dual_feas(cone)
    nt_point = cone.nt_point
    nt_point_sqrt = cone.nt_point_sqrt
    normalized_point = cone.normalized_point
    normalized_dual_point = cone.normalized_dual_point

    normalized_point .= cone.point ./ sqrt(cone.dist * 2) # NOTE the dist got scaled by 2
    normalized_dual_point .= cone.dual_point ./ sqrt(cone.dual_dist * 2)# NOTE the dist got scaled by 2
    gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)

    nt_point[1] = normalized_point[2] + normalized_dual_point[1]
    nt_point[2] = normalized_point[1] + normalized_dual_point[2]
    @. @views nt_point[3:end] = -normalized_point[3:end] + normalized_dual_point[3:end]
    nt_point ./= 2 * gamma
    @show 2 * nt_point[1] * nt_point[2] - sum(abs2, nt_point[3:end])

    copyto!(nt_point_sqrt, nt_point)
    nt_point_sqrt[1] += 1
    nt_point_sqrt ./= sqrt(2 * nt_point_sqrt[1])

    cone.rt_dist_ratio = sqrt(cone.dist / cone.dual_dist)
    cone.rt_rt_dist_ratio = sqrt(cone.rt_dist_ratio)

    cone.nt_updated = true

    return
end
