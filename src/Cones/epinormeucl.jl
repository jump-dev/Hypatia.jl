#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(u^2 - norm_2(w)^2) TODO barrier is modified

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
    dual_feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    hess_fact_updated::Bool
    scal_hess_updated::Bool
    inv_hess_updated::Bool
    nt_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    correction::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    normalized_point::Vector{T} # TODO can this and normalized_dual_point be removed?
    normalized_dual_point::Vector{T}
    nt_point::Vector{T} # actually a normalized NT point
    nt_point_sqrt::Vector{T}
    scaled_point::Vector{T}

    dist::T
    dual_dist::T
    rt_dist_ratio::T # sqrt(dist / dual_dist)
    rt_rt_dist_ratio::T # sqrt(sqrt(dist / dual_dist))
    corretion::Vector{T}

    # TODO remove fact_cache when NT in
    function EpiNormEucl{T}(
        dim::Int;
        use_dual::Bool = false, # TODO self-dual so maybe remove this option/field?
        max_neighborhood::Real = default_max_neighborhood(),
        ) where {T <: Real}
        @assert !use_dual # TODO delete later
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        return cone
    end
end

use_heuristic_neighborhood(cone::EpiNormEucl) = false

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.dual_feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated =
    cone.hess_fact_updated = cone.scal_hess_updated = cone.nt_updated = false)

use_nt(::EpiNormEucl) = true

use_scaling(::EpiNormEucl) = true

use_correction(::EpiNormEucl) = true

# TODO only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.scaled_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.correction = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)

    cone.normalized_point = zeros(T, dim)
    cone.normalized_dual_point = zeros(T, dim)
    cone.nt_point = zeros(T, dim)
    cone.nt_point_sqrt = zeros(T, dim)
    cone.correction = zeros(T, dim)
    cone.nt_point[1] = 1
    cone.nt_point_sqrt[1] = 1
    cone.normalized_point[1] = 1
    cone.normalized_dual_point[1] = 1
    cone.scaled_point[1] = 1
    cone.dual_dist = 1
    cone.rt_dist_ratio = 1
    cone.rt_rt_dist_ratio = 1

    return
end

get_nu(cone::EpiNormEucl) = 1

function set_initial_point(arr::AbstractVector, cone::EpiNormEucl{T}) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormEucl)
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > 0
        w = view(cone.point, 2:cone.dim)
        cone.dist = abs2(u) - sum(abs2, w)
        cone.is_feas = (cone.dist > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true

    return cone.is_feas
end

function update_dual_feas(cone::EpiNormEucl)
    u = cone.dual_point[1]

    if u > 0
        w = view(cone.dual_point, 2:cone.dim)
        cone.dual_dist = abs2(u) - sum(abs2, w)
        dual_feas = (cone.dual_dist > 0)
    else
        dual_feas = false
    end
    cone.dual_feas_updated = true

    return dual_feas
end

function update_grad(cone::EpiNormEucl)
    @assert cone.is_feas

    @. cone.grad = cone.point / cone.dist
    cone.grad[1] *= -1

    cone.grad_updated = true
    return cone.grad
end

function update_scal_hess(cone::EpiNormEucl{T}, mu::T) where {T}
    @assert cone.grad_updated
    @assert cone.is_feas
    cone.nt_updated || update_nt(cone)

    # if cone.use_scaling
        # analogous to W as a function of nt_point_sqrt
        # H = (2 * J * nt_point * nt_point' * J - J) * constant
        mul!(cone.hess.data, cone.nt_point, cone.nt_point', 2, false)
        cone.hess.data[:, 1] *= -1
        cone.hess += I # TODO inefficient
        cone.hess.data[1, :] *= -1
        cone.hess.data ./= cone.rt_dist_ratio
    # else
    #     mul!(cone.hess.data, cone.grad, cone.grad', 2, false)
    #     cone.hess += inv(cone.dist) * I # TODO inefficient
    #     cone.hess[1, 1] -= 2 / cone.dist
    # end

    # mul!(cone.hess.data, cone.nt_point, cone.nt_point', 2, false)
    # @. @views cone.hess.data[1, 2:cone.dim] *= -1
    # cone.hess += I # TODO inefficient
    # cone.hess.data[1, 1] -= 2
    # cone.hess.data ./= cone.rt_dist_ratio

    cone.hess_updated = true
    return cone.hess
end

 # NOTE not used
function update_inv_scal_hess(cone::EpiNormEucl)
    @assert cone.is_feas
    cone.nt_updated || update_nt(cone)

    # if cone.use_scaling
        # Hinv = (2 * nt_point * nt_point' - J) * constant
        mul!(cone.inv_hess.data, cone.nt_point, cone.nt_point', 2, false)
        cone.inv_hess += I # TODO inefficient
        cone.inv_hess[1, 1] -= 2
        cone.inv_hess.data .*= cone.rt_dist_ratio
    # else
    #     mul!(cone.inv_hess.data, cone.point, cone.point', 2, false)
    #     cone.inv_hess += cone.dist * I
    #     cone.inv_hess[1, 1] -= 2 * cone.dist
    # end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated

    mul!(cone.hess.data, cone.grad, cone.grad', 2, false)
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

    mul!(cone.inv_hess.data, cone.point, cone.point', 2, false)
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
        ga = 2 * (dot(w, wj) - u * uj) / cone.dist
        prod[1, j] = -ga * u - uj
        @. prod[2:end, j] = ga * w + wj
    end
    @. prod ./= cone.dist

    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas

    @inbounds for j in 1:size(prod, 2)
        @views pa = 2 * dot(cone.point, arr[:, j])
        @. prod[:, j] = pa * cone.point
    end
    @. @views prod[1, :] -= cone.dist * arr[1, :]
    @. @views prod[2:end, :] += cone.dist * arr[2:end, :]

    return prod
end

function inv_scal_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    cone.nt_updated || update_nt(cone)

    hyperbolic_householder(prod, arr, cone.nt_point, cone.rt_dist_ratio, Winv = false)

    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormEucl{T}) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    rt2 = sqrt(T(2))
    dist = cone.dist
    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        @views wj = arr[2:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (u * uj - dotwwj) / dist
        wmulj = (dotwwj / urtdist - uj) / dist
        @. prod[2:end, j] = w * wmulj + wj / rtdist
    end

    return prod
end

function scal_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    cone.nt_updated || update_nt(cone)

    hyperbolic_householder(prod, arr, cone.nt_point_sqrt, cone.rt_rt_dist_ratio, Winv = true)

    return prod
end

# NOTE not used
function inv_scal_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    cone.nt_updated || update_nt(cone)

    hyperbolic_householder(prod, arr, cone.nt_point_sqrt, cone.rt_rt_dist_ratio, Winv = false)

    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormEucl{T}) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        @views wj = arr[2:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (u * uj + dotwwj)
        wmulj = (dotwwj / urtdist + uj)
        @. prod[2:end, j] = w * wmulj + wj * rtdist
    end

    return prod
end

jdot(x::AbstractVector, y::AbstractVector) = @views x[1] * y[1] - dot(x[2:end], y[2:end])

function correction(cone::EpiNormEucl, primal_dir::AbstractVector, dual_dir::AbstractVector)
    @assert cone.grad_updated
    corr = cone.correction
    point = cone.point

    jdot_p_s = jdot(point, primal_dir)
    @. corr = jdot_p_s * dual_dir
    dot_s_z = dot(primal_dir, dual_dir)
    dot_p_z = dot(point, dual_dir)
    corr[1] += dot_s_z * point[1] - dot_p_z * primal_dir[1]
    @. @views corr[2:end] += -dot_s_z * point[2:end] + dot_p_z * primal_dir[2:end]
    corr ./= cone.dist

    return corr
end

function correction2(cone::EpiNormEucl, primal_dir::AbstractVector, dual_dir::AbstractVector)
    @assert cone.grad_updated
    corr = cone.correction
    point = cone.point

    # TODO only using primal_dir, and no Hinv_z
    # jdot_p_s = jdot(point, primal_dir)
    # @. corr = jdot_p_s * dual_dir
    # dot_s_z = dot(primal_dir, dual_dir)
    # dot_p_z = dot(point, dual_dir)
    # corr[1] += dot_s_z * point[1] - dot_p_z * primal_dir[1]
    # @. @views corr[2:end] += -dot_s_z * point[2:end] + dot_p_z * primal_dir[2:end]
    # corr ./= cone.dist

    return corr
end


# function step_and_update_scaling(cone::EpiNormEucl{T}, primal_dir::AbstractVector, dual_dir::AbstractVector, step_size::T) where {T}
function update_nt(cone::EpiNormEucl{T}) where {T}
    @assert cone.feas_updated
    cone.dual_feas_updated || update_dual_feas(cone)
    nt_point = cone.nt_point
    nt_point_sqrt = cone.nt_point_sqrt
    normalized_point = cone.normalized_point
    normalized_dual_point = cone.normalized_dual_point

    # # TODO put this somehwere more appropriate, better to judge when scaled updates are decided upon
    # @views cone.dual_dist = abs2(cone.dual_point[1]) - sum(abs2, cone.dual_point[2:end])
    # @assert cone.dual_dist >= 0

    # if cone.try_scaled_updates
    #     # based on CVXOPT code
    #     # scale the primal and dual directions under old scaling, store results as points to be normalized
    #     hyperbolic_householder(normalized_point, primal_dir, nt_point_sqrt, cone.rt_rt_dist_ratio, Winv = true)
    #     hyperbolic_householder(normalized_dual_point, dual_dir, nt_point_sqrt, cone.rt_rt_dist_ratio)
    #     # get new primal/dual points but in the old scaling
    #     scaled_point = cone.scaled_point
    #     @. normalized_point *= step_size
    #     @. normalized_dual_point *= step_size
    #     @. normalized_point += scaled_point
    #     @. normalized_dual_point += scaled_point
    #
    #     # normalize
    #     primal_dir_dist = distnorm(normalized_point)
    #     dual_dir_dist = distnorm(normalized_dual_point)
    #     normalized_point ./= primal_dir_dist
    #     normalized_dual_point ./= dual_dir_dist
    #
    #     gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)
    #     vs = dot(nt_point_sqrt, normalized_point)
    #     vz = jdot(nt_point_sqrt, normalized_dual_point)
    #     vq = (vs + vz) / 2 / gamma
    #     vu = vs - vz
    #     scaled_point[1] = gamma
    #
    #     w1 = 2 * nt_point_sqrt[1] * vq - (normalized_point[1] + normalized_dual_point[1] ) / 2 / gamma
    #     d = (nt_point_sqrt[1] * vu - (normalized_point[1] - normalized_dual_point[1]) / 2) / (w1 + 1)
    #
    #     # updates the scaled point
    #     @views copyto!(scaled_point[2:end], nt_point_sqrt[2:end])
    #     @. scaled_point[2:end] *= (vu - 2 * d * vq)
    #     @. scaled_point[2:end] += normalized_point[2:end] * (1 - d / gamma) / 2
    #     @. scaled_point[2:end] += normalized_dual_point[2:end] * (1 + d / gamma) / 2
    #     @. scaled_point *= sqrt(primal_dir_dist * dual_dir_dist)
    #
    #     # updates the NT scaling point
    #     @. nt_point = 2 * vq * nt_point_sqrt
    #     nt_point[1] -= normalized_point[1] / 2 / gamma
    #     @. @views nt_point[2:end] += normalized_point[2:end] / 2 / gamma
    #     @. nt_point -= normalized_dual_point / gamma / 2
    #
    #     # updates the squareroot of the NT scaling point
    #     copyto!(nt_point_sqrt, nt_point)
    #     nt_point_sqrt[1] += 1
    #     @. nt_point_sqrt /= sqrt(2 * nt_point_sqrt[1])
    #
    #     cone.rt_dist_ratio *= primal_dir_dist / dual_dir_dist
    #     cone.rt_rt_dist_ratio *= sqrt(primal_dir_dist / dual_dir_dist)
    # else
        # section 4 CVXOPT paper / part of ECOS' update each iteration
        normalized_point .= cone.point ./ sqrt(cone.dist)
        normalized_dual_point .= cone.dual_point ./ sqrt(cone.dual_dist)
        gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)

        nt_point[1] = normalized_point[1] + normalized_dual_point[1]
        @. @views nt_point[2:end] = normalized_point[2:end] - normalized_dual_point[2:end]
        nt_point ./= 2 * gamma

        copyto!(nt_point_sqrt, nt_point)
        nt_point_sqrt[1] += 1
        nt_point_sqrt ./= sqrt(2 * nt_point_sqrt[1])

        cone.rt_dist_ratio = sqrt(cone.dist / cone.dual_dist)
        cone.rt_rt_dist_ratio = sqrt(cone.rt_dist_ratio)

        cone.nt_updated = true
    # end

    return
end

function hyperbolic_householder(prod::AbstractVecOrMat, arr::AbstractVecOrMat, v::AbstractVector, fact::Real; Winv::Bool = false)
    if Winv
        v[2:end] *= -1
    end
    for j in 1:size(prod, 2)
        @views pa = 2 * dot(v, arr[:, j])
        @. @views prod[:, j] = pa * v
    end
    @. prod[1, :] -= arr[1, :]
    @. prod[2:end, :] += arr[2:end, :]
    if Winv
        prod ./= fact
        v[2:end] *= -1
    else
        prod .*= fact
    end
    return prod
end
