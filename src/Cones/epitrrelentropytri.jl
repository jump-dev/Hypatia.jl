#=
epigraph of the relative entropy cone
(u in R, V in S_+^d, W in S_+^d) : u >= tr(W * log(W) - W * log(V))

derivatives for quantum relative entropy function adapted from
"Long-Step Path-Following Algorithm in Quantum Information Theory: Some Numerical Aspects and Applications"
by L. Faybusovich and C. Zhou

uses the log-homogeneous but not self-concordant barrier
-log(u - tr(W * log(W) - W * log(V))) - logdet(W) - logdet(V)

TODO reduce allocations
=#

mutable struct EpiTrRelEntropyTri{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    correction::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    rt2::T
    V::Matrix{T}
    W::Matrix{T}
    Vi::Matrix{T}
    Wi::Matrix{T}
    V_idxs::UnitRange{Int}
    W_idxs::UnitRange{Int}
    vw_dim::Int
    z::T
    dzdV::Vector{T} # divided by z
    dzdW::Vector{T}
    W_similar::Matrix{T}
    mat::Matrix{T}
    matsdim1::Matrix{T}
    matsdim2::Matrix{T}
    tempsdim::Vector{T}
    diff_mat_V::Matrix{T}
    diff_mat_W::Matrix{T}
    diff_tensor_V::Array{T, 3}
    V_fact
    W_fact
    V_vals_log::Vector{T}
    W_vals_log::Vector{T}
    V_log::Matrix{T}
    W_log::Matrix{T}
    WV_log::Matrix{T}
    dz_sqr_dV_sqr::Matrix{T}
    dz_sqr_dW_sqr::Matrix{T}
    dz_sqr_dW_dV::Matrix{T}

    function EpiTrRelEntropyTri{T}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim > 2
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.vw_dim = div(dim - 1, 2)
        cone.d = round(Int, sqrt(0.25 + 2 * cone.vw_dim) - 0.5)
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    dim = cone.dim
    vw_dim = cone.vw_dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    d = cone.d
    cone.rt2 = sqrt(T(2))
    cone.V = zeros(T, d, d)
    cone.W = zeros(T, d, d)
    cone.Vi = zeros(T, d, d)
    cone.Wi = zeros(T, d, d)
    cone.V_idxs = 2:(vw_dim + 1)
    cone.W_idxs = (vw_dim + 2):cone.dim
    cone.dzdV = zeros(T, vw_dim)
    cone.dzdW = zeros(T, vw_dim)
    cone.W_similar = zeros(T, d, d)
    cone.mat = zeros(T, d, d)
    cone.matsdim1 = zeros(T, vw_dim, vw_dim)
    cone.matsdim2 = zeros(T, vw_dim, vw_dim)
    cone.tempsdim = zeros(T, vw_dim)
    cone.diff_mat_V = zeros(T, d, d)
    cone.diff_mat_W = zeros(T, d, d)
    cone.diff_tensor_V = zeros(T, d, d, d)
    cone.V_vals_log = zeros(T, d)
    cone.W_vals_log = zeros(T, d)
    cone.V_log = zeros(T, d, d)
    cone.W_log = zeros(T, d, d)
    cone.WV_log = zeros(T, d, d)
    cone.dz_sqr_dV_sqr = zeros(T, vw_dim, vw_dim)
    cone.dz_sqr_dW_sqr = zeros(T, vw_dim, vw_dim)
    cone.dz_sqr_dW_dV = zeros(T, vw_dim, vw_dim)
    return
end

get_nu(cone::EpiTrRelEntropyTri) = 2 * cone.d + 1

function set_initial_point(arr::AbstractVector, cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    arr .= 0
    # at the initial point V and W are diagonal, equivalent to epirelentropy
    (arr[1], v, w) = get_central_ray_epirelentropy(cone.d)
    k = 1
    for i in 1:cone.d
        arr[1 + k] = v
        arr[cone.vw_dim + 1 + k] = w
        k += i + 1
    end
    return arr
end

function update_feas(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    @assert !cone.feas_updated
    point = cone.point
    vw_dim = cone.vw_dim
    @views V = Hermitian(svec_to_smat!(cone.V, point[cone.V_idxs], cone.rt2), :U)
    @views W = Hermitian(svec_to_smat!(cone.W, point[cone.W_idxs], cone.rt2), :U)

    cone.is_feas = false
    (V_vals, V_vecs) = cone.V_fact = eigen(V)
    if isposdef(cone.V_fact)
        (W_vals, W_vecs) = cone.W_fact = eigen(W)
        if isposdef(cone.W_fact)
            @. cone.V_vals_log = log(V_vals)
            @. cone.W_vals_log = log(W_vals)
            mul!(cone.mat, V_vecs, Diagonal(cone.V_vals_log))
            V_log = mul!(cone.V_log, cone.mat, V_vecs')
            mul!(cone.mat, W_vecs, Diagonal(cone.W_vals_log))
            W_log = mul!(cone.W_log, cone.mat, W_vecs')
            @. cone.WV_log = W_log - V_log
            cone.z = point[1] - dot(W, Hermitian(cone.WV_log, :U))
            cone.is_feas = (cone.z > 0)
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(::EpiTrRelEntropyTri) = true

function update_grad(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    @assert cone.is_feas
    d = cone.d
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    W = Hermitian(cone.W, :U)
    z = cone.z
    (V_vals, V_vecs) = cone.V_fact
    (W_vals, W_vecs) = cone.W_fact
    Vi = cone.Vi = inv(cone.V_fact)
    Wi = cone.Wi = inv(cone.W_fact)

    cone.grad[1] = -inv(z)

    @views smat_to_svec!(cone.grad[W_idxs], Wi, rt2)
    dzdW = smat_to_svec!(cone.dzdW, -(cone.WV_log + I) / z, rt2)
    @. @views cone.grad[W_idxs] += dzdW
    @. @views cone.grad[W_idxs] *= -1

    diff_mat_V = cone.diff_mat_V
    diff_mat!(diff_mat_V, V_vals, cone.V_vals_log)
    W_similar = cone.W_similar = V_vecs' * W * V_vecs
    temp = V_vecs * (W_similar .* Hermitian(diff_mat_V, :U)) * V_vecs' / z
    grad_V = -temp - Vi
    @views smat_to_svec!(cone.grad[V_idxs], grad_V, rt2)
    @views smat_to_svec!(cone.dzdV, temp, rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiTrRelEntropyTri{T}) where {T <: Real}
    @assert cone.grad_updated
    d = cone.d
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    z = cone.z
    (V_vals, V_vecs) = cone.V_fact
    (W_vals, W_vecs) = cone.W_fact
    Vi = cone.Vi
    Wi = cone.Wi
    H = cone.hess.data

    diff_mat!(cone.diff_mat_W, W_vals, cone.W_vals_log)
    diff_mat_V = Hermitian(cone.diff_mat_V, :U)
    diff_mat_W = Hermitian(cone.diff_mat_W, :U)
    diff_tensor_V = diff_tensor!(cone.diff_tensor_V, diff_mat_V, V_vals)

    W_similar = cone.W_similar
    dz_sqr_dV_sqr = hess_tr_logm!(cone.dz_sqr_dV_sqr, V_vecs, W_similar, diff_tensor_V, rt2)
    @. dz_sqr_dV_sqr *= -1
    @views Hvv = H[V_idxs, V_idxs]
    symm_kron(Hvv, Vi, rt2)
    dzdV = cone.dzdV
    mul!(Hvv, dzdV, dzdV', true, true)
    @. Hvv += dz_sqr_dV_sqr / z

    dz_sqr_dW_sqr = grad_logm!(cone.dz_sqr_dW_sqr, W_vecs, cone.matsdim1, cone.matsdim2, cone.tempsdim, diff_mat_W, rt2)
    @views Hww = H[W_idxs, W_idxs]
    symm_kron(Hww, Wi, rt2)
    dzdW = cone.dzdW
    mul!(Hww, dzdW, dzdW', true, true)
    @. Hww += dz_sqr_dW_sqr / z

    dz_sqr_dW_dV = grad_logm!(cone.dz_sqr_dW_dV, V_vecs, cone.matsdim1, cone.matsdim2, cone.tempsdim, diff_mat_V, rt2)
    @views Hvw = H[V_idxs, W_idxs]
    @. Hvw = -dz_sqr_dW_dV / z
    mul!(Hvw, dzdV, dzdW', true, true)

    H[1, 1] = -cone.grad[1]
    @views H[1, V_idxs] .= dzdV
    @views H[1, W_idxs] .= dzdW
    @views H[1, :] ./= z

    cone.hess_updated = true
    return cone.hess
end

function correction(cone::EpiTrRelEntropyTri{T}, dir::AbstractVector{T}) where T
    @assert cone.hess_updated
    d = cone.d
    corr = cone.correction
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    z = cone.z
    (V_vals, V_vecs) = cone.V_fact
    (W_vals, W_vecs) = cone.W_fact
    Vi = Hermitian(cone.Vi)
    Wi = Hermitian(cone.Wi)
    dzdV = cone.dzdV * z
    dzdW = cone.dzdW * z
    diff_mat_V = Hermitian(cone.diff_mat_V, :U)
    diff_mat_W = Hermitian(cone.diff_mat_W, :U)
    diff_tensor_V = cone.diff_tensor_V
    W_similar = cone.W_similar
    dz_sqr_dV_sqr = cone.dz_sqr_dV_sqr
    dlogW_dW = cone.dz_sqr_dW_sqr
    dlogV_dV = cone.dz_sqr_dW_dV

    u_dir = dir[1]
    @views v_dir = dir[V_idxs]
    @views w_dir = dir[W_idxs]
    @views V_dir = Symmetric(svec_to_smat!(zeros(T, d, d), dir[V_idxs], cone.rt2), :U)
    @views W_dir = Symmetric(svec_to_smat!(zeros(T, d, d), dir[W_idxs], cone.rt2), :U)
    V_dir_similar = V_vecs' * V_dir * V_vecs
    W_dir_similar = W_vecs' * W_dir * W_vecs

    diff_tensor_W = zeros(T, d, d, d)
    diff_tensor!(diff_tensor_W, diff_mat_W, W_vals)

    VW_dir_similar = V_vecs' * W_dir * V_vecs
    diff_dot_V_VV = [V_dir_similar[:, q]' * Diagonal(diff_tensor_V[:, p, q]) * V_dir_similar[:, p] for p in 1:d, q in 1:d]
    d2logV_dV2_VV = V_vecs * diff_dot_V_VV * V_vecs'
    diff_dot_V_VW = [V_dir_similar[:, q]' * Diagonal(diff_tensor_V[:, p, q]) * VW_dir_similar[:, p] for p in 1:d, q in 1:d]
    diff_dot_W_WW = [W_dir_similar[:, q]' * Diagonal(diff_tensor_W[:, p, q]) * W_dir_similar[:, p] for p in 1:d, q in 1:d]

    dlogV_dV_dw = Symmetric(svec_to_smat!(zeros(T, d, d), dlogV_dV * w_dir, cone.rt2), :U)
    dlogV_dV_dv = Symmetric(svec_to_smat!(zeros(T, d, d), dlogV_dV * v_dir, cone.rt2), :U)
    dlogW_dW_dw = Symmetric(svec_to_smat!(zeros(T, d, d), dlogW_dW * w_dir, cone.rt2), :U)
    dz_sqr_dV_sqr_dv = Symmetric(svec_to_smat!(zeros(T, d, d), dz_sqr_dV_sqr * v_dir, cone.rt2), :U)
    const0 = (u_dir + dot(dzdV, v_dir) + dot(dzdW, w_dir)) / z
    const1 = abs2(const0) + (dot(v_dir, dz_sqr_dV_sqr, v_dir) / 2 + dot(w_dir, dlogW_dW, w_dir) / 2 - dot(v_dir, dlogV_dV, w_dir)) / z

    # u
    corr[1] = const1

    # v
    diff_quad = zeros(T, cone.vw_dim^2, cone.vw_dim^2)
    diff_quad!(diff_quad, diff_tensor_V, V_vals)
    d3WlogVdV = zeros(T, d, d)
    for j in 1:d, i in 1:d, l in 1:d, k in 1:d
        d3WlogVdV[i, j] += diff_quad[block_idxs(d, l)[k], block_idxs(d, j)[i]] * (
            V_dir_similar[k, l] * V_dir_similar[l, j] * W_similar[i, k] +
            V_dir_similar[k, l] * V_dir_similar[l, i] * W_similar[j, k] +
            V_dir_similar[k, i] * V_dir_similar[l, j] * W_similar[k, l]
            )
    end
    V_part_1 = (dz_sqr_dV_sqr_dv - dlogV_dV_dw) * const0
    sqrt_vals = sqrt.(V_vals)
    rdiv!(V_dir_similar, Diagonal(sqrt_vals))
    ldiv!(Diagonal(V_vals), V_dir_similar)
    V_part_2a = V_dir_similar * V_dir_similar'
    V_part_2 = V_vecs * (diff_dot_V_VW + diff_dot_V_VW' + d3WlogVdV + V_part_2a * z) * V_vecs'
    V_corr = V_part_1 + V_part_2
    @views smat_to_svec!(corr[V_idxs], V_corr, cone.rt2)
    @. @views corr[V_idxs] += dzdV * const1

    # w
    W_part_1 = const0 * (dlogW_dW_dw - dlogV_dV_dv) + d2logV_dV2_VV
    sqrt_vals = sqrt.(W_vals)
    rdiv!(W_dir_similar, Diagonal(sqrt_vals))
    ldiv!(Diagonal(W_vals), W_dir_similar)
    W_part_2a = W_dir_similar * W_dir_similar'
    W_part_2 = W_vecs * (W_part_2a * z - diff_dot_W_WW) * W_vecs'
    W_corr = W_part_1 + W_part_2
    @views smat_to_svec!(corr[W_idxs], W_corr, cone.rt2)
    @. @views corr[W_idxs] += dzdW * const1

    @. corr /= z

    return corr
end

function hess_tr_logm!(mat, vecs, mat_inner, diff_tensor, rt2::T) where T
    d = size(vecs, 1)
    X = zeros(T, d, d, d)
    for i in 1:d
        X[i, :, :] = vecs * (diff_tensor[i, :, :] .* mat_inner) * vecs'
    end
    temp = Symmetric(zeros(T, d^2, d^2), :U)
    for j in 1:d, i in 1:j
        temp.data[block_idxs(d, i), block_idxs(d, j)] = vecs * Diagonal(X[:, i, j]) * vecs'
    end

    mat .= 0
    row_idx = 1
    for j in 1:d, i in 1:j
        col_idx = 1
        for l in 1:d, k in 1:l
            mat[row_idx, col_idx] +=
                (
                temp[block_idxs(d, j), block_idxs(d, l)][i, k] +
                temp[block_idxs(d, i), block_idxs(d, l)][j, k] +
                temp[block_idxs(d, j), block_idxs(d, k)][i, l] +
                temp[block_idxs(d, i), block_idxs(d, k)][j, l]
                ) * (i == j ? 1 : rt2) * (k == l ? 1 : rt2)
            col_idx += 1
        end
        row_idx += 1
    end
    mat ./= 2

    return mat
end

function diff_quad!(diff_quad::Matrix{T}, diff_tensor::Array{T, 3}, V_vals::Vector{T}) where T
    rteps = sqrt(eps(T))
    d = length(V_vals)
    idx1 = 1
    @inbounds for j in 1:d, i in 1:d
        idx2 = 1
        for l in 1:d, k in 1:d
            (vi, vj, vk, vl) = (V_vals[i], V_vals[j], V_vals[k], V_vals[l])
            if (abs(vi - vj) < rteps) && (abs(vi - vk) < rteps) && (abs(vi - vl) < rteps)
                t = inv(vi^3) / 3 # fourth derivative divided by 3!
            elseif (abs(vi - vl) < rteps) && (abs(vi - vk) < rteps)
                t = (diff_tensor[i, i, i] - diff_tensor[i, i, j]) / (vi - vj)
            elseif (abs(vi - vl) < rteps)
                t = (diff_tensor[i, i, j] - diff_tensor[i, j, k]) / (vi - vk)
            else
                t = (diff_tensor[j, k, l] - diff_tensor[i, j, k]) / (vl - vi)
            end
            diff_quad[idx1, idx2] = t
            idx2 += 1
        end
        idx1 += 1
    end
    return diff_quad
end
