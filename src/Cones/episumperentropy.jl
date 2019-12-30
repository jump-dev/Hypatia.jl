#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) epigraph of sum of perspectives of entropies (AKA vector relative entropy cone)
(u in R, v in R_+^n, w in R_+^n) : u >= sum_i w_i*log(w_i/v_i)

barrier from "Primal-Dual Interior-Point Methods for Domain-Driven Formulations" by Karimi & Tuncel, 2019
-log(u - sum_i w_i*log(w_i/v_i)) - sum_i (log(v_i) + log(w_i))
=#

mutable struct EpiSumPerEntropy{T <: Real} <: Cone{T}
    use_dual::Bool
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

    v_idxs::UnitRange{Int}
    w_idxs::UnitRange{Int}
    diff::T

    function EpiSumPerEntropy{T}(
        dim::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual = is_dual
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
    return
end

get_nu(cone::EpiSumPerEntropy) = cone.dim

function set_initial_point(arr::AbstractVector, cone::EpiSumPerEntropy)
    # (arr[1], v, w) = get_central_ray_episumperentropy(cone.w_dim) # TODO if needed
    arr[1] = 1
    arr[cone.v_idxs] .= 1
    arr[cone.w_idxs] .= 1
    return arr
end

function update_feas(cone::EpiSumPerEntropy)
    @assert !cone.feas_updated
    u = cone.point[1]

    cone.is_feas = false
    if u > 0
        @views v = cone.point[cone.v_idxs]
        @views w = cone.point[cone.w_idxs]
        if all(vi -> vi > 0, w) && all(wi -> wi > 0, w)
            wlwv = sum(wi * log(wi / vi) for (vi, wi) in zip(v, w))
            cone.diff = u - wlwv
            cone.is_feas = (cone.diff > 0)
        end
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

    g[1] = -inv(diff)
    @. g[cone.v_idxs] = (-w / diff - 1) / v
    @. g[cone.w_idxs] = -inv(w) + (log(w / v) + 1) / diff # TODO reuse log(w/v) from feas

    cone.grad_updated = true
    return cone.grad
end

# (1/(u - w1 log(w1/v1) - w2 log(w2/v2))^2
# w1/(v1 (u - w1 log(w1/v1) - w2 log(w2/v2))^2)
# w2/(v2 (u - w1 log(w1/v1) - w2 log(w2/v2))^2)
# (-log(w1/v1) - 1)/(u - w1 log(w1/v1) - w2 log(w2/v2))^2
# (-log(w2/v2) - 1)/(u - w1 log(w1/v1) - w2 log(w2/v2))^2
#
# w1^2/(v1^2 (u - w1 log(w1/v1) - w2 log(w2/v2))^2) + w1/(v1^2 (u - w1 log(w1/v1) - w2 log(w2/v2))) + 1/v1^2
# (w1 w2)/(v1 v2 (u - w1 log(w1/v1) - w2 log(w2/v2))^2)
# (w1 (-log(w1/v1) - 1))/(v1 (u - w1 log(w1/v1) - w2 log(w2/v2))^2) - 1/(v1 (u - w1 log(w1/v1) - w2 log(w2/v2)))
# (w1 (-log(w2/v2) - 1))/(v1 (u - w1 log(w1/v1) - w2 log(w2/v2))^2)
#
# w2^2/(v2^2 (u - w1 log(w1/v1) - w2 log(w2/v2))^2) + w2/(v2^2 (u - w1 log(w1/v1) - w2 log(w2/v2))) + 1/v2^2
# (w2 (-log(w1/v1) - 1))/(v2 (u - w1 log(w1/v1) - w2 log(w2/v2))^2)
# (w2 (-log(w2/v2) - 1))/(v2 (u - w1 log(w1/v1) - w2 log(w2/v2))^2) - 1/(v2 (u - w1 log(w1/v1) - w2 log(w2/v2)))
#
# (-log(w1/v1) - 1)^2/(u - w1 log(w1/v1) - w2 log(w2/v2))^2 + 1/(w1 (u - w1 log(w1/v1) - w2 log(w2/v2))) + 1/w1^2
# ((-log(w1/v1) - 1) (-log(w2/v2) - 1))/(u - w1 log(w1/v1) - w2 log(w2/v2))^2
#
# (-log(w2/v2) - 1)^2/(u - w1 log(w1/v1) - w2 log(w2/v2))^2 + 1/(w2 (u - w1 log(w1/v1) - w2 log(w2/v2))) + 1/w2^2)

# TODO improve efficiency and numerics and style
function update_hess(cone::EpiSumPerEntropy)
    @assert cone.grad_updated
    w_dim = cone.w_dim
    u = cone.point[1]
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]
    diff = cone.diff
    g = cone.grad
    g1 = g[1]
    H = cone.hess.data

    # H_u_u, H_u_v, H_u_w parts
    H[1, 1] = abs2(g1)
    @. H[1, v_idxs] = w / v / diff / diff
    @. H[1, w_idxs] = -(log(w / v) + 1) / diff / diff # TODO reuse from g

    # H_v_v, H_v_w, H_w_w parts
    for (i, v_idx, w_idx) in zip(1:w_dim, v_idxs, w_idxs)
        vi = point[v_idx]
        wi = point[w_idx]
        H[v_idx, v_idx] = (abs2(wi / diff) + wi / diff + 1) / vi / vi # TODO calc wi / diff once
        H[v_idx, w_idx] = (wi * (-log(wi / vi) - 1)) / (vi * diff^2) - inv(diff) / vi
        H[w_idx, w_idx] = (-log(wi / vi) - 1)^2 / diff^2 + inv(diff) / wi + inv(wi)^2

        for j in 1:(i - 1)
            v_idx2 = v_idxs[j]
            w_idx2 = w_idxs[j]
            vj = point[v_idx2]
            wj = point[w_idx2]
            H[v_idx, w_idx2] = wi * (-log(wj / vj) - 1) / vi / diff / diff
        end

        for j in (i + 1):w_dim
            v_idx2 = v_idxs[j]
            w_idx2 = w_idxs[j]
            vj = point[v_idx2]
            wj = point[w_idx2]
            H[v_idx, v_idx2] = (wi * wj) / (vi * vj * diff^2) # TODO faster as wi / vi / diff * (..)
            H[v_idx, w_idx2] = wi * (-log(wj / vj) - 1) / vi / diff / diff
            H[w_idx, w_idx2] = (-log(wi / vi) - 1) * (-log(wj / vj) - 1) / diff / diff
        end
    end

    cone.hess_updated = true
    return cone.hess
end
