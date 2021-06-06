"""
$(TYPEDEF)

Epigraph of vector relative entropy cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int, use_dual::Bool = false)
"""
mutable struct EpiRelEntropy{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    w_dim::Int
    v_idxs::UnitRange{Int64}
    w_idxs::UnitRange{Int64}

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_aux_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, SparseMatrixCSC{T, Int}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    lwv::Vector{T}
    tau::Vector{T}
    sigma::Vector{T}
    z::T
    Hiuu::T
    Hiuv::Vector{T}
    Hiuw::Vector{T}
    Hivv::Vector{T}
    Hivw::Vector{T}
    Hiww::Vector{T}
    temp1::Vector{T}
    temp2::Vector{T}

    function EpiRelEntropy{T}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real}
        @assert dim >= 3
        @assert isodd(dim)
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        w_dim = cone.w_dim = div(dim - 1, 2)
        cone.v_idxs = 1 .+ (1:w_dim)
        cone.w_idxs = w_dim .+ cone.v_idxs
        return cone
    end
end

reset_data(cone::EpiRelEntropy) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_aux_updated =
    cone.hess_fact_updated = false)

function setup_extra_data!(cone::EpiRelEntropy{T}) where {T <: Real}
    w_dim = cone.w_dim
    cone.lwv = zeros(T, w_dim)
    cone.tau = zeros(T, w_dim)
    cone.sigma = zeros(T, w_dim)
    cone.Hiuv = zeros(T, w_dim)
    cone.Hiuw = zeros(T, w_dim)
    cone.Hivv = zeros(T, w_dim)
    cone.Hivw = zeros(T, w_dim)
    cone.Hiww = zeros(T, w_dim)
    cone.temp1 = zeros(T, w_dim)
    cone.temp2 = zeros(T, w_dim)
    return cone
end

get_nu(cone::EpiRelEntropy) = cone.dim

function set_initial_point!(arr::AbstractVector, cone::EpiRelEntropy)
    (arr[1], v, w) = get_central_ray_epirelentropy(cone.w_dim)
    @views arr[cone.v_idxs] .= v
    @views arr[cone.w_idxs] .= w
    return arr
end

function update_feas(cone::EpiRelEntropy{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    @views v = cone.point[cone.v_idxs]
    @views w = cone.point[cone.w_idxs]

    if all(>(eps(T)), v) && all(>(eps(T)), w)
        @. cone.lwv = log(w / v)
        cone.z = u - dot(w, cone.lwv)
        cone.is_feas = (cone.z > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiRelEntropy{T}) where T
    u = cone.dual_point[1]
    @views v = cone.dual_point[cone.v_idxs]
    @views w = cone.dual_point[cone.w_idxs]

    if all(>(eps(T)), v) && u > eps(T)
        return all(u * (1 + log(vi / u)) + wi > eps(T) for (vi, wi) in zip(v, w))
    end

    return false
end

function update_grad(cone::EpiRelEntropy)
    @assert cone.is_feas
    @views v = cone.point[cone.v_idxs]
    @views w = cone.point[cone.w_idxs]
    z = cone.z
    sigma = cone.sigma
    tau = cone.tau
    g = cone.grad

    @. sigma = w / v / z
    @. tau = (cone.lwv + 1) / -z
    g[1] = -inv(z)
    @. @views g[cone.v_idxs] = -sigma - inv(v)
    @. @views g[cone.w_idxs] = -tau - inv(w)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiRelEntropy{T}) where T
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    u = cone.point[1]
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]
    z = cone.z
    sigma = cone.sigma
    tau = cone.tau
    g = cone.grad

    # H_u_u, H_u_v, H_u_w parts
    H[1, 1] = abs2(g[1])
    @. @views H[1, v_idxs] = sigma / z
    @. @views H[1, w_idxs] = tau / z

    # H_v_v, H_v_w, H_w_w parts
    zinv = inv(z)
    @inbounds for (i, v_idx, w_idx) in zip(1:cone.w_dim, v_idxs, w_idxs)
        vi = point[v_idx]
        wi = point[w_idx]
        taui = tau[i]
        sigmai = sigma[i]

        H[v_idx, v_idx] = abs2(sigmai) - g[v_idx] / vi
        H[w_idx, w_idx] = abs2(taui) + (zinv + inv(wi)) / wi

        @. H[v_idx, w_idxs] = sigmai * tau
        @. H[w_idx, v_idxs] = sigma * taui
        H[v_idx, w_idx] -= zinv / vi

        @inbounds for j in (i + 1):cone.w_dim
            H[v_idx, v_idxs[j]] = sigmai * sigma[j]
            H[w_idx, w_idxs[j]] = taui * tau[j]
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# auxiliary calculations for inverse Hessian
function update_inv_hess_aux(cone::EpiRelEntropy{T}) where T
    @assert !cone.inv_hess_aux_updated
    point = cone.point
    @views v = point[cone.v_idxs]
    @views w = point[cone.w_idxs]
    z = cone.z

    HiuHu = zero(T)
    @inbounds for i in 1:cone.w_dim
        wi = w[i]
        vi = v[i]
        lwvi = cone.lwv[i]
        zwi = z + wi
        z2wi = zwi + wi
        wz2wi = wi / z2wi
        vz2wi = vi / z2wi
        uvvi = wi * (wi * lwvi - z)
        uwwi = wi * (z + lwvi * zwi) * wz2wi
        HiuHu += wz2wi * uvvi - uwwi * (lwvi + 1)
        cone.Hiuv[i] = vz2wi * uvvi
        cone.Hiuw[i] = uwwi
        cone.Hivw[i] = wi * vi * wz2wi
        cone.Hiww[i] = wi * zwi * wz2wi
        cone.Hivv[i] = vi * zwi * vz2wi
    end
    cone.Hiuu = abs2(z) - HiuHu
    if cone.Hiuu < zero(T)
        @warn("bad Hiuu $(cone.Hiuu)")
    end

    cone.inv_hess_aux_updated = true
    return
end

function alloc_inv_hess!(cone::EpiRelEntropy{T}) where T
    # initialize sparse idxs for upper triangle of inverse Hessian
    dim = cone.dim
    w_dim = cone.w_dim
    nnz_tri = 2 * dim - 1 + w_dim
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    @views I[idxs1] .= 1
    @views J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    @views I[idxs2] .= 2:dim
    @views J[idxs2] .= 2:dim
    idxs3 = (2 * dim):nnz_tri
    @views I[idxs3] .= 1 .+ (1:w_dim)
    @views J[idxs3] .= (1 + w_dim) .+ (1:w_dim)
    V = ones(T, nnz_tri)
    cone.inv_hess = Symmetric(sparse(I, J, V, dim, dim), :U)
    return
end

# updates for nonzero values in the inverse Hessian
function update_inv_hess(cone::EpiRelEntropy{T}) where T
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    w_dim = cone.w_dim

    # modify nonzeros of upper triangle of inverse Hessian
    nzval = cone.inv_hess.data.nzval
    nzval[1] = cone.Hiuu
    nz_idx = 2
    @inbounds for i in 1:cone.w_dim
        nzval[nz_idx] = cone.Hiuv[i]
        nzval[nz_idx + 1] = cone.Hivv[i]
        nz_idx += 2
    end
    @inbounds for i in 1:cone.w_dim
        nzval[nz_idx] = cone.Hiuw[i]
        nzval[nz_idx + 1] = cone.Hivw[i]
        nzval[nz_idx + 2] = cone.Hiww[i]
        nz_idx += 3
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiRelEntropy,
    )
    @assert cone.grad_updated
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    @views v = cone.point[v_idxs]
    @views w = cone.point[w_idxs]
    z = cone.z
    sigma = cone.sigma
    tau = cone.tau

    @inbounds @views begin
        u_arr = arr[1, :]
        u_prod = prod[1, :]
        v_arr = arr[v_idxs, :]
        v_prod = prod[v_idxs, :]
        w_arr = arr[w_idxs, :]
        w_prod = prod[w_idxs, :]
        mul!(u_prod, v_arr', sigma)
        mul!(u_prod, w_arr', tau, true, true)
        @. u_prod += u_arr / z
        @. v_prod = sigma * u_prod' + (sigma * v_arr + v_arr / v - w_arr / z) / v
        @. w_prod = tau * u_prod' + (w_arr / z + w_arr / w) / w - v_arr / v / z
        @. u_prod /= z
    end

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiRelEntropy,
    )
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)

    @inbounds @views begin
        u_arr = arr[1, :]
        u_prod = prod[1, :]
        v_arr = arr[cone.v_idxs, :]
        v_prod = prod[cone.v_idxs, :]
        w_arr = arr[cone.w_idxs, :]
        w_prod = prod[cone.w_idxs, :]
        @. u_prod = cone.Hiuu * u_arr
        mul!(u_prod, v_arr', cone.Hiuv, true, true)
        mul!(u_prod, w_arr', cone.Hiuw, true, true)
        mul!(v_prod, cone.Hiuv, u_arr')
        mul!(w_prod, cone.Hiuw, u_arr')
        @. v_prod += cone.Hivv * v_arr + cone.Hivw * w_arr
        @. w_prod += cone.Hivw * v_arr + cone.Hiww * w_arr
    end

    return prod
end

function dder3(cone::EpiRelEntropy, dir::AbstractVector)
    @assert cone.grad_updated
    tau = cone.tau
    z = cone.z
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    @views v = cone.point[v_idxs]
    @views w = cone.point[w_idxs]
    u_dir = dir[1]
    @views v_dir = dir[v_idxs]
    @views w_dir = dir[w_idxs]
    dder3 = cone.dder3
    @views v_dder3 = dder3[v_idxs]
    @views w_dder3 = dder3[w_idxs]
    wdw = cone.temp1
    vdv = cone.temp2

    i2z = inv(2 * z)
    @. wdw = w_dir / w
    @. vdv = v_dir / v
    const0 = (u_dir + dot(w, vdv)) / z + dot(tau, w_dir)
    const1 = abs2(const0) + sum(w[i] * abs2(vdv[i]) + w_dir[i] *
        (wdw[i] - 2 * vdv[i]) for i in eachindex(w)) / (2 * z)
    dder3[1] = const1 / z

    # v
    v_dder3 .= const1
    @. v_dder3 += (const0 + vdv) * vdv - i2z * wdw * w_dir
    @. v_dder3 *= w
    @. v_dder3 += (z * vdv - w_dir) * vdv + (-const0 + i2z * w_dir) * w_dir
    @. v_dder3 /= v
    @. v_dder3 /= z

    # w
    @. w_dder3 = const1 * tau
    @. w_dder3 += ((const0 - w * vdv / z) / z + (inv(w) + i2z) * wdw) * wdw
    @. w_dder3 += (-const0 + w_dir / z - vdv / 2) / z * vdv

    return dder3
end

# TODO remove this in favor of new hess_nz_count etc functions that directly use uu, uw, ww etc
inv_hess_nz_count(cone::EpiRelEntropy) =
    3 * cone.dim - 2 + 2 * cone.w_dim

inv_hess_nz_count_tril(cone::EpiRelEntropy) =
    2 * cone.dim - 1 + cone.w_dim

inv_hess_nz_idxs_col(cone::EpiRelEntropy, j::Int) =
    (j == 1 ? (1:cone.dim) : (j <= (1 + cone.w_dim) ?
    [1, j, j + cone.w_dim] : [1, j - cone.w_dim, j]))

inv_hess_nz_idxs_col_tril(cone::EpiRelEntropy, j::Int) =
    (j == 1 ? (1:cone.dim) : (j <= (1 + cone.w_dim) ? [j, j + cone.w_dim] : [j]))

# see analysis in
# https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_epirelentropy(w_dim::Int)
    if w_dim <= 10
        return central_rays_epirelentropy[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    rtw_dim = sqrt(w_dim)
    if w_dim <= 20
        u = 1.2023 / rtw_dim - 0.015
        v = 0.432 / rtw_dim + 1.0125
        w = -0.3057 / rtw_dim + 0.972
    else
        u = 1.1513 / rtw_dim - 0.0069
        v = 0.4873 / rtw_dim + 1.0008
        w = -0.4247 / rtw_dim + 0.9961
    end
    return [u, v, w]
end

const central_rays_epirelentropy = [
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
