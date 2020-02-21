#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) epigraph of sum of perspectives of entropies (AKA vector relative entropy cone)
(u in R, v in R_+^n, w in R_+^n) : u >= sum_i w_i*log(w_i/v_i)

barrier from "Primal-Dual Interior-Point Methods for Domain-Driven Formulations" by Karimi & Tuncel, 2019
-log(u - sum_i w_i*log(w_i/v_i)) - sum_i (log(v_i) + log(w_i))

TODO
- write native tests for use_dual = true
=#

mutable struct EpiSumPerEntropy{T <: Real} <: Cone{T}
    use_dual::Bool
    max_neighborhood::T
    dim::Int
    w_dim::Int
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
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    v_idxs::UnitRange{Int}
    w_idxs::UnitRange{Int}
    lwv1d::Vector{T}
    diff::T
    wvdiff::Vector{T}

    function EpiSumPerEntropy{T}(
        dim::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual = is_dual
        cone.max_neighborhood = default_max_neighborhood()
        cone.dim = dim
        cone.w_dim = div(dim - 1, 2)
        cone.v_idxs = 2:(cone.w_dim + 1)
        cone.w_idxs = (cone.w_dim + 2):dim
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

EpiSumPerEntropy{T}(dim::Int) where {T <: Real} = EpiSumPerEntropy{T}(dim, false)

# TODO only allocate the fields we use
function setup_data(cone::EpiSumPerEntropy{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.lwv1d = zeros(T, cone.w_dim)
    cone.wvdiff = zeros(T, cone.w_dim)
    return
end

get_nu(cone::EpiSumPerEntropy) = cone.dim

function set_initial_point(arr::AbstractVector, cone::EpiSumPerEntropy)
    (arr[1], v, w) = get_central_ray_episumperentropy(div(cone.dim - 1, 2))
    arr[cone.v_idxs] .= v
    arr[cone.w_idxs] .= w
    return arr
end

function update_feas(cone::EpiSumPerEntropy)
    @assert !cone.feas_updated
    u = cone.point[1]
    @views v = cone.point[cone.v_idxs]
    @views w = cone.point[cone.w_idxs]

    if all(vi -> vi > 0, v) && all(wi -> wi > 0, w)
        @. cone.lwv1d = log(w / v)
        cone.diff = u - dot(w, cone.lwv1d)
        cone.is_feas = (cone.diff > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiSumPerEntropy)
    @assert cone.is_feas
    u = cone.point[1]
    @views v = cone.point[cone.v_idxs]
    @views w = cone.point[cone.w_idxs]
    diff = cone.diff
    g = cone.grad
    lwv1d = cone.lwv1d

    @. lwv1d += 1
    @. lwv1d /= -diff
    g[1] = -inv(diff)
    @. g[cone.v_idxs] = (-w / diff - 1) / v
    @. g[cone.w_idxs] = -inv(w) - lwv1d

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiSumPerEntropy)
    @assert cone.grad_updated
    w_dim = cone.w_dim
    u = cone.point[1]
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]
    lwv1d = cone.lwv1d
    diff = cone.diff
    wvdiff = cone.wvdiff
    H = cone.hess.data

    # H_u_u, H_u_v, H_u_w parts
    H[1, 1] = abs2(cone.grad[1])
    @. wvdiff = w / v / diff
    @. H[1, v_idxs] = wvdiff / diff
    @. H[1, w_idxs] = lwv1d / diff

    # H_v_v, H_v_w, H_w_w parts
    @inbounds for (i, v_idx, w_idx) in zip(1:w_dim, v_idxs, w_idxs)
        vi = point[v_idx]
        wi = point[w_idx]
        lwv1di = lwv1d[i]
        wvdiffi = wvdiff[i]
        invvi = inv(vi)

        H[v_idx, v_idx] = abs2(wvdiffi) + (wvdiffi + invvi) / vi
        H[w_idx, w_idx] = abs2(lwv1di) + (inv(diff) + inv(wi)) / wi

        @. H[v_idx, w_idxs] = wvdiffi * lwv1d
        H[v_idx, w_idx] -= invvi / diff

        @inbounds for j in (i + 1):w_dim
            H[v_idx, v_idxs[j]] = wvdiffi * wvdiff[j]
            H[w_idx, w_idxs[j]] = lwv1di * lwv1d[j]
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_episumperentropy(w_dim::Int)
    if w_dim <= 10
        # lookup points where x = f'(x)
        return central_rays_episumperentropy[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    if w_dim <= 20
        u = 1.2023 / sqrt(w_dim) - 0.015
        v = 0.432 / sqrt(w_dim) + 1.0125
        w = -0.3057 / sqrt(w_dim) + 0.972
    else
        u = 1.1513 / sqrt(w_dim) - 0.0069
        v = 0.4873 / sqrt(w_dim) + 1.0008
        w = -0.4247 / sqrt(w_dim) + 0.9961
    end
    return [u, v, w]
end

const central_rays_episumperentropy = [
    0.827838399	1.290927714	0.805102005;
    0.708612491	1.256859155	0.818070438;
    0.622618845	1.231401008	0.829317079;
    0.558111266	1.211710888	0.838978357;
    0.508038611	1.196018952	0.847300431;
    0.468039614	1.183194753	0.854521307;
    0.435316653	1.172492397	0.860840992;
    0.408009282	1.163403374	0.866420017;
    0.38483862	1.155570329	0.871385499;
    0.364899122	1.148735192	0.875838068;
    ]
