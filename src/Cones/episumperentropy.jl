#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) epigraph of sum of perspectives of entropies (AKA vector relative entropy cone)
(u in R, v in R_+^n, w in R_+^n) : u >= sum_i w_i*log(w_i/v_i) TODO update description here for non-contiguous v/w

barrier from "Primal-Dual Interior-Point Methods for Domain-Driven Formulations" by Karimi & Tuncel, 2019
-log(u - sum_i w_i*log(w_i/v_i)) - sum_i (log(v_i) + log(w_i))

TODO
- write native tests for use_dual = true
=#

mutable struct EpiSumPerEntropy{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    w_dim::Int
    v_idxs
    w_idxs
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_aux_updated::Bool
    hess_sqrt_aux_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, SparseMatrixCSC{T, Int}}
    inv_hess_sqrt::UpperTriangular{T, SparseMatrixCSC{T, Int}}
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    lwv::Vector{T}
    tau::Vector{T}
    z::T
    Hiu::Vector{T}
    Hiuu::T
    Hivv::Vector{T}
    Hivw::Vector{T}
    Hiww::Vector{T}
    temp1::Vector{T}
    temp2::Vector{T}

    function EpiSumPerEntropy{T}(
        dim::Int;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.w_dim = div(dim - 1, 2)
        cone.v_idxs = 2:2:(dim - 1)
        cone.w_idxs = 3:2:dim
        return cone
    end
end

reset_data(cone::EpiSumPerEntropy) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated = cone.hess_sqrt_aux_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiSumPerEntropy{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    w_dim = cone.w_dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.lwv = zeros(T, w_dim)
    cone.tau = zeros(T, w_dim)
    cone.Hiu = zeros(T, dim - 1)
    cone.Hivv = zeros(T, w_dim)
    cone.Hivw = zeros(T, w_dim)
    cone.Hiww = zeros(T, w_dim)
    cone.temp1 = zeros(T, w_dim)
    cone.temp2 = zeros(T, w_dim)
    return
end

use_correction(cone::EpiSumPerEntropy) = true

get_nu(cone::EpiSumPerEntropy) = cone.dim

function set_initial_point(arr::AbstractVector, cone::EpiSumPerEntropy)
    (arr[1], v, w) = get_central_ray_episumperentropy(cone.w_dim)
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
        @. cone.lwv = log(w / v)
        cone.z = u - dot(w, cone.lwv)
        cone.is_feas = (cone.z > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_dual_feas(cone::EpiSumPerEntropy{T}) where {T <: Real}
    u = cone.dual_point[1]
    @views v = cone.dual_point[cone.v_idxs]
    @views w = cone.dual_point[cone.w_idxs]
    if all(vi -> vi > 0, v) && u > 0
        return all(u * (1 + log(vi / u)) + wi > 0 for (vi, wi) in zip(v, w))
    end
    return false
end

function update_grad(cone::EpiSumPerEntropy)
    @assert cone.is_feas
    u = cone.point[1]
    @views v = cone.point[cone.v_idxs]
    @views w = cone.point[cone.w_idxs]
    z = cone.z
    g = cone.grad
    tau = cone.tau

    @. tau = (cone.lwv + 1) / -z
    g[1] = -inv(z)
    @. g[cone.v_idxs] = (-w / z - 1) / v
    @. g[cone.w_idxs] = -inv(w) - tau

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiSumPerEntropy{T}) where {T}
    @assert cone.grad_updated
    if !isdefined(cone, :hess)
        cone.hess = Symmetric(zeros(T, cone.dim, cone.dim), :U)
    end
    u = cone.point[1]
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]
    tau = cone.tau
    z = cone.z
    sigma = cone.temp1
    H = cone.hess.data

    # H_u_u, H_u_v, H_u_w parts
    H[1, 1] = abs2(cone.grad[1])
    @. sigma = w / v / z
    @. H[1, v_idxs] = sigma / z
    @. H[1, w_idxs] = tau / z

    # H_v_v, H_v_w, H_w_w parts
    zi = inv(z)
    @inbounds for (i, v_idx, w_idx) in zip(1:cone.w_dim, v_idxs, w_idxs)
        vi = point[v_idx]
        wi = point[w_idx]
        taui = tau[i]
        sigmai = sigma[i]
        invvi = inv(vi)

        H[v_idx, v_idx] = abs2(sigmai) + (sigmai + invvi) / vi
        H[w_idx, w_idx] = abs2(taui) + (zi + inv(wi)) / wi

        @. H[v_idx, w_idxs] = sigmai * tau
        @. H[w_idx, v_idxs] = sigma * taui
        H[v_idx, w_idx] -= invvi / z

        @inbounds for j in (i + 1):cone.w_dim
            H[v_idx, v_idxs[j]] = sigmai * sigma[j]
            H[w_idx, w_idxs[j]] = taui * tau[j]
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# auxiliary calculations for inverse Hessian
function update_hess_aux(cone::EpiSumPerEntropy{T}) where {T}
    @assert !cone.hess_aux_updated
    u = cone.point[1]
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]
    z = cone.z
    Hiu = cone.Hiu
    denom = cone.temp1
    @. denom = z + 2 * w

    HiuHu = zero(T)
    @inbounds for (i, v_idx, w_idx) in zip(1:cone.w_dim, v_idxs, w_idxs)
        lwv = cone.lwv[i]
        wi = w[i]
        vi = v[i]
        wlwv = wi * lwv
        scal = wi / (z + 2 * wi)
        Hiuv = Hiu[v_idx - 1] = vi * (wlwv - z) * scal
        Hiuw = Hiu[w_idx - 1] = wi * (z * (lwv + 1) + wlwv) * scal
        HiuHu += Hiuv * wi / vi + Hiuw * cone.tau[i] * z
    end

    cone.Hiuu = abs2(z) - HiuHu
    @assert cone.Hiuu > 0
    @. cone.Hivw = v * abs2(w) / denom
    @. cone.Hivv = (z + w) / denom
    @. cone.Hiww = cone.Hivv * abs2(w)
    @. cone.Hivv *= abs2(v)

    cone.hess_aux_updated = true
    return
end

function sparse_upper_arrow_block2(T::Type{<:Real}, w_dim::Int)
    dim = 2 * w_dim + 1
    nnz_tri = 2 * dim - 1 + w_dim
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    I[idxs1] .= 1
    J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    I[idxs2] .= 2:dim
    J[idxs2] .= 2:dim
    idxs3 = (2 * dim):nnz_tri
    I[idxs3] .= 2:2:dim
    J[idxs3] .= 3:2:dim
    V = ones(T, nnz_tri)
    return sparse(I, J, V, dim, dim)
end

# updates for nonzero values in the inverse Hessian
function update_inv_hess(cone::EpiSumPerEntropy{T}) where {T}
    cone.hess_aux_updated || update_hess_aux(cone)

    if !isdefined(cone, :inv_hess)
        # initialize sparse idxs for upper triangle of inverse Hessian
        cone.inv_hess = Symmetric(sparse_upper_arrow_block2(T, cone.w_dim), :U)
    end

    # modify nonzeros of upper triangle of inverse Hessian
    nzval = cone.inv_hess.data.nzval
    nzval[1] = cone.Hiuu
    nz_idx = 2
    dim_idx = 1
    @inbounds for i in 1:cone.w_dim
        nzval[nz_idx] = cone.Hiu[dim_idx]
        nzval[nz_idx + 1] = cone.Hivv[i]
        nz_idx += 2
        dim_idx += 1
        nzval[nz_idx] = cone.Hiu[dim_idx]
        nzval[nz_idx + 1] = cone.Hivw[i]
        nzval[nz_idx + 2] = cone.Hiww[i]
        nz_idx += 3
        dim_idx += 1
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiSumPerEntropy{T}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)

    Hivv = cone.Hivv
    Hivw = cone.Hivw
    Hiww = cone.Hiww
    @. @views prod[1, :] = arr[1, :] * cone.Hiuu
    @views mul!(prod[1, :], arr[2:end, :]', cone.Hiu, true, true)
    @. @views prod[2:end, :] = cone.Hiu * arr[1, :]'
    @inbounds for i in 1:cone.w_dim
        vi = 2i
        wi = vi + 1
        @views arrv = arr[vi, :]
        @views arrw = arr[wi, :]
        @. prod[vi, :] += Hivv[i] * arrv + Hivw[i] * arrw
        @. prod[wi, :] += Hiww[i] * arrw + Hivw[i] * arrv
    end

    return prod
end

# auxiliary calculations for sqrt prod and hess prod oracles
function update_hess_sqrt_aux(cone::EpiSumPerEntropy{T}) where {T}
    cone.hess_aux_updated || update_hess_aux(cone)
    @assert !cone.hess_sqrt_aux_updated

    if !isdefined(cone, :inv_hess_sqrt)
        # initialize sparse idxs for upper triangular factor of inverse Hessian
        cone.inv_hess_sqrt = UpperTriangular(sparse_upper_arrow_block2(T, cone.w_dim))
    end

    # modify nonzeros of upper triangular factor of inverse Hessian
    Hiu = cone.Hiu
    Hivv = cone.Hivv
    Hivw = cone.Hivw
    Hiww = cone.Hiww
    nzval = cone.inv_hess_sqrt.data.nzval
    nz_idx = 1
    uu2 = cone.Hiuu
    minrt = eps(T)
    @inbounds for i in 1:cone.w_dim
        vi = 2i
        ww = sqrt(max(Hiww[i], minrt))
        vw = Hivw[i] / ww
        uw = Hiu[vi] / ww
        vv = sqrt(max(Hivv[i] - abs2(vw), minrt))
        uv = (Hiu[vi - 1] - vw * uw) / vv
        uu2 -= abs2(uw) + abs2(uv)
        @. nzval[nz_idx .+ (1:5)] = (uv, vv, uw, vw, ww)
        nz_idx += 5
    end
    nzval[1] = sqrt(max(uu2, minrt))

    cone.hess_sqrt_aux_updated = true
    return
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiSumPerEntropy)
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    ldiv!(cone.inv_hess_sqrt', ldiv!(prod, cone.inv_hess_sqrt, arr))
    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiSumPerEntropy)
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    copyto!(prod, arr)
    lmul!(cone.inv_hess_sqrt', prod)
    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiSumPerEntropy)
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    ldiv!(prod, cone.inv_hess_sqrt, arr)
    return prod
end

function correction2(cone::EpiSumPerEntropy{T}, primal_dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    tau = cone.tau
    z = cone.z
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    @views v = cone.point[v_idxs]
    @views w = cone.point[w_idxs]
    u_dir = primal_dir[1]
    v_dir = primal_dir[v_idxs]
    w_dir = primal_dir[w_idxs]
    corr = cone.correction
    v_corr = view(corr, v_idxs)
    w_corr = view(corr, w_idxs)
    wdw = cone.temp1
    vdv = cone.temp2

    i2z = inv(2 * z)
    @. wdw = w_dir / w
    @. vdv = v_dir / v
    const0 = (u_dir + dot(w, vdv)) / z + dot(tau, w_dir)
    const1 = abs2(const0) + sum(w[i] * abs2(vdv[i]) + w_dir[i] * (wdw[i] - 2 * vdv[i]) for i in eachindex(w)) / (2 * z)
    corr[1] = const1 / z

    # v
    v_corr .= const1
    @. v_corr += (const0 + vdv) * vdv - i2z * wdw * w_dir
    @. v_corr *= w
    @. v_corr += (z * vdv - w_dir) * vdv + (-const0 + i2z * w_dir) * w_dir
    @. v_corr /= v
    @. v_corr /= z

    # w
    @. w_corr = const1 * tau
    @. w_corr += ((const0 - w * vdv / z) / z + (inv(w) + i2z) * wdw) * wdw
    @. w_corr += (-const0 + w_dir / z - vdv / 2) / z * vdv

    return corr
end

# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_episumperentropy(w_dim::Int)
    if w_dim <= 10
        return central_rays_episumperentropy[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    rtwdim = sqrt(w_dim)
    if w_dim <= 20
        u = 1.2023 / rtwdim - 0.015
        v = 0.432 / rtwdim + 1.0125
        w = -0.3057 / rtwdim + 0.972
    else
        u = 1.1513 / rtwdim - 0.0069
        v = 0.4873 / rtwdim + 1.0008
        w = -0.4247 / rtwdim + 0.9961
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
