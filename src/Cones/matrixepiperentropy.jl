#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

derivatives for quantum relative entropy function adapted from
"Long-Step Path-Following Algorithm in Quantum Information Theory: Some Numerical Aspects and Applications"
by L. Faybusovich and C. Zhou


TODO
replace eigen decomposition
initial point
=#

import GenericLinearAlgebra

# TODO hack around https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl/issues/51 while using AD
function logm(A)
    (vals, vecs) = GenericLinearAlgebra.eigen(Hermitian(A))
    # vals = Complex.(vals)
    return vecs * Diagonal(log.(vals)) * vecs'
end

function grad_logm(V_vecs, diff_mat, sdim, side)
    ret = zeros(sdim, sdim)
    row_idx = 1
    for j in 1:side, i in 1:j
        col_idx = 1
        for l in 1:side, k in 1:l
            ret[row_idx, col_idx] += sum(diff_mat[m, n] * (
                V_vecs[i, m] * V_vecs[k, m] * V_vecs[l, n] * V_vecs[j, n] +
                V_vecs[j, m] * V_vecs[k, m] * V_vecs[l, n] * V_vecs[i, n] +
                V_vecs[i, m] * V_vecs[l, m] * V_vecs[k, n] * V_vecs[j, n] +
                V_vecs[j, m] * V_vecs[l, m] * V_vecs[k, n] * V_vecs[i, n]
                ) * (m == n ? 1 : 2) * (i == j ? 1 : sqrt(2)) * (k == l ? 1 : sqrt(2)) / 4
                for m in 1:side for n in 1:m)
            col_idx += 1
        end
        row_idx += 1
    end
    return ret
end

function hess_tr_logm(V_vecs, W_similar, diff_tensor_V, sdim, side)
    ret = zeros(sdim, sdim)
    row_idx = 1
    for j in 1:side, i in 1:j
        col_idx = 1
        for l in 1:side, k in 1:l
            ret[row_idx, col_idx] += sum(
                (
                V_vecs[i, m] * V_vecs[j, n] * (V_vecs[k, m] * dot(V_vecs[l, :], W_similar[:, n] .* diff_tensor_V[m, n, :]) + V_vecs[l, n] * dot(V_vecs[k, :], W_similar[:, m] .* diff_tensor_V[m, n, :])) +
                V_vecs[j, m] * V_vecs[i, n] * (V_vecs[k, m] * dot(V_vecs[l, :], W_similar[:, n] .* diff_tensor_V[m, n, :]) + V_vecs[l, n] * dot(V_vecs[k, :], W_similar[:, m] .* diff_tensor_V[m, n, :])) +
                V_vecs[i, m] * V_vecs[j, n] * (V_vecs[l, m] * dot(V_vecs[k, :], W_similar[:, n] .* diff_tensor_V[m, n, :]) + V_vecs[k, n] * dot(V_vecs[l, :], W_similar[:, m] .* diff_tensor_V[m, n, :])) +
                V_vecs[j, m] * V_vecs[i, n] * (V_vecs[l, m] * dot(V_vecs[k, :], W_similar[:, n] .* diff_tensor_V[m, n, :]) + V_vecs[k, n] * dot(V_vecs[l, :], W_similar[:, m] .* diff_tensor_V[m, n, :]))
                ) *
                (m == n ? 1 : 2) * (i == j ? 1 : sqrt(2)) * (k == l ? 1 : sqrt(2)) / 4
                for m in 1:side for n in 1:m)
            col_idx += 1
        end
        row_idx += 1
    end
    return ret
end

mutable struct MatrixEpiPerEntropy{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    rt2::T
    side::Int
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}
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
    correction::Vector{T} # TODO
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    V
    W
    Vi
    Wi
    V_idxs
    W_idxs
    vw_dim
    z
    dzdV
    dzdW
    W_similar
    tmp
    diff_mat
    V_fact
    W_fact

    function MatrixEpiPerEntropy{T}(
        dim::Int;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim > 1
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.vw_dim = div(dim - 1, 2)
        cone.side = round(Int, sqrt(0.25 + 2 * cone.vw_dim) - 0.5)
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

# TODO only allocate the fields we use
function setup_data(cone::MatrixEpiPerEntropy{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.rt2 = sqrt(T(2))
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    side = cone.side
    cone.V = zeros(T, side, side)
    cone.W = zeros(T, side, side)
    cone.V_idxs = 2:(cone.vw_dim + 1)
    cone.W_idxs = (cone.vw_dim + 2):cone.dim
    cone.dzdV = zeros(T, cone.vw_dim)
    cone.dzdW = zeros(T, cone.vw_dim)
    cone.W_similar = zeros(T, side, side)
    cone.tmp = zeros(T, side, side)
    cone.diff_mat = zeros(T, side, side)
    return
end

get_nu(cone::MatrixEpiPerEntropy) = 2 * cone.side + 1

use_correction(::MatrixEpiPerEntropy) = false

function set_initial_point(arr::AbstractVector, cone::MatrixEpiPerEntropy{T}) where {T <: Real}
    arr .= 0
    k = 1
    for i in 1:cone.side
        arr[1 + k] = 1
        arr[cone.vw_dim + 1 + k] = 1
        k += i + 1
    end
    arr[1] = 1
    return arr
end

function update_feas(cone::MatrixEpiPerEntropy{T}) where {T <: Real}
    @assert !cone.feas_updated
    point = cone.point
    vw_dim = cone.vw_dim
    @views V = Hermitian(svec_to_smat!(cone.V, point[2:(vw_dim + 1)], cone.rt2), :U)
    @views W = Hermitian(svec_to_smat!(cone.W, point[(vw_dim + 2):end], cone.rt2), :U)
    if isposdef(V) && isposdef(W)
        cone.z = point[1] - tr(W * logm(W) - W * logm(V))
        cone.is_feas = (cone.z > 0)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::MatrixEpiPerEntropy) = true # TODO use a dikin ellipsoid condition?

function update_grad(cone::MatrixEpiPerEntropy{T}) where {T <: Real}
    @assert cone.is_feas
    side = cone.side
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    V = Hermitian(svec_to_smat!(cone.V, cone.point[V_idxs], rt2), :U)
    W = Hermitian(svec_to_smat!(cone.W, cone.point[W_idxs], rt2), :U)
    z = cone.z
    (V_vals, V_vecs) = cone.V_fact = eigen(V)
    (W_vals, W_vecs) = cone.W_fact = eigen(W)
    Vi = cone.Vi = V_vecs * Diagonal(inv.(V_vals)) * V_vecs'
    Wi = cone.Wi = W_vecs * Diagonal(inv.(W_vals)) * W_vecs'

    cone.grad[1] = -inv(z)

    dzdW = logm(V) - logm(W) - I
    grad_W = -dzdW / z - Wi
    @views smat_to_svec!(cone.grad[W_idxs], grad_W, rt2)

    diff_mat = cone.diff_mat
    for j in 1:side, i in 1:j
        (vi, vj) = (V_vals[i], V_vals[j])
        if abs(vi - vj) < sqrt(eps(T))
            diff_mat[i, j] = inv(vi)
        else
            diff_mat[i, j] = (log(vi) - log(vj)) / (vi - vj)
        end
    end
    W_similar = cone.W_similar = V_vecs' * W * V_vecs
    tmp = -V_vecs * (W_similar .* Hermitian(diff_mat, :U)) * V_vecs' / z
    dzdV = @views smat_to_svec!(cone.dzdV, tmp, rt2)
    grad_V = tmp - Vi
    @views smat_to_svec!(cone.grad[V_idxs], grad_V, rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::MatrixEpiPerEntropy{T}) where {T <: Real}
    @assert cone.is_feas
    side = cone.side
    rt2 = cone.rt2
    V_idxs = cone.V_idxs
    W_idxs = cone.W_idxs
    V = Hermitian(cone.V, :U)
    W = Hermitian(cone.W, :U)
    z = cone.z
    vw_dim = cone.vw_dim
    (V_vals, V_vecs) = cone.V_fact
    (W_vals, W_vecs) = cone.W_fact
    Vi = cone.Vi
    Wi = cone.Wi
    H = cone.hess.data

    diff_mat_V = zeros(side, side)
    diff_mat_W = zeros(side, side)
    for j in 1:side, i in 1:j
        (vi, vj) = (V_vals[i], V_vals[j])
        (wi, wj) = (W_vals[i], W_vals[j])
        if abs(vi - vj) < sqrt(eps(T))
            diff_mat_V[i, j] = inv(vi)
        else
            diff_mat_V[i, j] = (log(vi) - log(vj)) / (vi - vj)
        end
        if abs(wi - wj) < sqrt(eps(T))
            diff_mat_W[i, j] = inv(wi)
        else
            diff_mat_W[i, j] = (log(wi) - log(wj)) / (wi - wj)
        end
    end
    diff_mat_V = Hermitian(diff_mat_V, :U)
    diff_mat_W = Hermitian(diff_mat_W, :U)

    diff_tensor_V = zeros(side, side, side)
    for k in 1:side, j in 1:k, i in 1:j
        (vi, vj, vk) = (V_vals[i], V_vals[j], V_vals[k])
        if abs(vj - vk) < sqrt(eps())
            if abs(vi - vj) < sqrt(eps())
                diff_tensor_V[i, i, i] = -inv(vi) / vi / 2
            else
                diff_tensor_V[i, j, j] = diff_tensor_V[j, i, j] = diff_tensor_V[j, j, i] = -(inv(vj) - diff_mat_V[i, j]) / (vi - vj)
            end
        elseif abs(vi - vj) < sqrt(eps())
            diff_tensor_V[k, j, j] = diff_tensor_V[j, k, j] = diff_tensor_V[j, j, k] = (inv(vi) - diff_mat_V[k, i]) / (vi - vk)
        else
            diff_tensor_V[i, j, k] = diff_tensor_V[i, k, j] = diff_tensor_V[j, i, k] =
                diff_tensor_V[j, k, i] = diff_tensor_V[k, i, j] = diff_tensor_V[k, j, i] = (diff_mat_V[i, j] - diff_mat_V[k, i]) / (vj - vk)
        end
    end

    W_similar = cone.W_similar
    dz_sqr_dV_sqr = hess_tr_logm(V_vecs, W_similar, diff_tensor_V, vw_dim, side)
    dzdV = cone.dzdV
    dz_dV_sqr = dzdV * dzdV'
    ViVi = symm_kron(zeros(vw_dim, vw_dim), Vi, rt2)
    Hvv = dz_dV_sqr - dz_sqr_dV_sqr / z + ViVi

    dz_sqr_dW_sqr = grad_logm(W_vecs, diff_mat_W, vw_dim, side)
    dz_dW = logm(V) - logm(W) - I
    dz_dW_vec = smat_to_svec!(zeros(vw_dim), dz_dW, rt2)
    dz_dW_sqr = dz_dW_vec * dz_dW_vec'
    WiWi = symm_kron(zeros(vw_dim, vw_dim), Wi, rt2)
    Hww = dz_dW_sqr / z / z + dz_sqr_dW_sqr / z + WiWi

    # TODO more annoying to think about but possibly simplier to compute from dzdv
    dz_sqr_dW_dV = grad_logm(V_vecs, diff_mat_V, vw_dim, side)
    dz_dW_dz_dV = dz_dW_vec * dzdV'
    Hwv = -dz_sqr_dW_dV / z - dz_dW_dz_dV / z

    Huu = inv(z) / z
    Huv = -dzdV / z
    Huw = dz_dW_vec / z / z

    H[1, 1] = inv(z) / z
    @views H[1, V_idxs] .= Huv
    @views H[1, W_idxs] .= Huw
    @views H[V_idxs, V_idxs] .= Hvv
    @views H[V_idxs, W_idxs] .= Hwv'
    @views H[W_idxs, W_idxs] .= Hww

    cone.hess_updated = true
    return cone.hess
end
