#=
(closure of) epigraph of perspective of the quantum entropy function
(u in R, v in R_+, W in S_+^d) : u >= tr(W * log(W / v))

barrier -log(u - tr(W * log(W / v))) - log(v) - logdet(W) where log() is the matrix logarithm
=#

mutable struct EpiPerTraceEntropyTri{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    rt2::T

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

    W::Matrix{T}
    Wi::Matrix{T}
    fact_W
    z::T
    lwv::Matrix{T}
    tau::Matrix{T}
    sigma::T

    function EpiPerTraceEntropyTri{T}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.d = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

use_heuristic_neighborhood(::EpiPerTraceEntropyTri) = false

use_correction(::EpiPerTraceEntropyTri) = false

function setup_extra_data(cone::EpiPerTraceEntropyTri{T}) where T
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.lwv = zeros(T, cone.d, cone.d)
    cone.tau = zeros(T, cone.d, cone.d)
    cone.W = zeros(T, cone.d, cone.d)
    return cone
end

get_nu(cone::EpiPerTraceEntropyTri) = cone.d + 2

function set_initial_point(arr::AbstractVector{T}, cone::EpiPerTraceEntropyTri{T}) where T
    d = cone.d
    (u, w) = get_central_ray_epiperentropy(d)
    arr[1] = u
    arr[2] = sqrt(T(d) * w * u + one(T))
    for i in 1:d
        arr[2 + sum(1:i)] = w
    end
    return arr
end

function update_feas(cone::EpiPerTraceEntropyTri{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    @views W = Symmetric(svec_to_smat!(cone.W, cone.point[3:cone.dim], cone.rt2), :U)

    (w_vals, w_vecs) = eigen(W)

    if (v > eps(T)) && all(wi -> wi > eps(T), w_vals)
        cone.lwv = w_vecs * Diagonal(log.(w_vals ./ v)) * w_vecs'
        # cone.lwv = log_W - cone.d * log(v)
        cone.z = u - dot(W, Symmetric(cone.lwv))
        cone.is_feas = (cone.z > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiPerTraceEntropyTri{T}) where T
    u = cone.dual_point[1]
    v = cone.dual_point[2]
    @views W = Symmetric(svec_to_smat!(similar(cone.point, cone.d, cone.d), cone.dual_point[3:cone.dim], cone.rt2), :U)
    if u > eps(T)
        return v - u * tr(exp.(-W ./ u - I)) > eps(T)
    end

    return false
end

function update_grad(cone::EpiPerTraceEntropyTri)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    W = Symmetric(cone.W, :U)
    Wi = cone.Wi = inv(W)
    z = cone.z
    cone.sigma = tr(W) / v / z
    cone.tau = (cone.lwv + I) / z
    tau = cone.tau
    sigma = cone.sigma

    cone.grad[1] = -inv(z)
    cone.grad[2] = -sigma - inv(v)
    @views smat_to_svec!(cone.grad[3:end], tau - Wi, cone.rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerTraceEntropyTri{T}) where T
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    W = Symmetric(cone.W, :U)
    Wi = Symmetric(cone.Wi, :U)
    z = cone.z
    tau = cone.tau
    sigma = cone.sigma
    H = cone.hess.data

    (w_vals, w_vecs) = eigen(W)
    d = cone.d
    diff_mat = zeros(d, d)
    for j in 1:d, i in 1:d
        (vi, vj) = (w_vals[i], w_vals[j])
        if abs(vi - vj) < sqrt(eps())
            diff_mat[i, j] = inv(vi)
        else
            diff_mat[i, j] = (log(vi) - log(vj)) / (vi - vj)
        end
    end
    diff_mat = Hermitian(diff_mat, :U)

    H[1, 1] = abs2(inv(z))
    H[1, 2] = sigma / z
    @views smat_to_svec!(H[1, 3:end], -tau / z, cone.rt2)
    H[2, 2] = abs2(sigma) + sigma / v + inv(v) / v
    @views smat_to_svec!(H[2, 3:end], -sigma * tau - I / v / z, cone.rt2)
    @views symm_kron(H[3:end, 3:end], Wi, cone.rt2)
    tau_vec = smat_to_svec!(zeros(cone.dim - 2), tau, cone.rt2)
    @views mul!(H[3:end, 3:end], tau_vec, tau_vec', true, true)
    Hww = @views H[3:end, 3:end]

    idx = 1
    for j in 1:cone.d, i in 1:j
        Eij = zeros(d, d)
        Eij[i, j] = (i == j ? 1 : inv(cone.rt2))
        Eij[j, i] = (i == j ? 1 : inv(cone.rt2))
        Eij_similar = w_vecs' * Eij * w_vecs
        res = w_vecs * (Eij_similar .* diff_mat) * w_vecs' / z #* (i == j ? 1 : cone.rt2)
        @views smat_to_svec_add!(Hww[idx, :], res, cone.rt2)
        idx += 1
    end

    # row_idx = 1
    # for j in 1:cone.d, i in 1:j
    #     col_idx = 1
    #     for l in 1:cone.d, k in 1:l
    #         Hww[row_idx, col_idx] += sum(diff_mat[m, n] / z * (
    #             w_vecs[i, m] * w_vecs[k, m] * w_vecs[l, n] * w_vecs[j, n] +
    #             w_vecs[j, m] * w_vecs[k, m] * w_vecs[l, n] * w_vecs[i, n] +
    #             w_vecs[i, m] * w_vecs[l, m] * w_vecs[k, n] * w_vecs[j, n] +
    #             w_vecs[j, m] * w_vecs[l, m] * w_vecs[k, n] * w_vecs[i, n] +
    #             0
    #             ) * (m == n ? 1 : 2) * (i == j ? 1 : cone.rt2) * (k == l ? 1 : cone.rt2) / 4
    #             for n in 1:cone.d for m in 1:n)
    #         col_idx += 1
    #     end
    #     row_idx += 1
    # end

    # idx1 = 1
    # for j in 1:cone.d, i in 1:j
    #     idx2 = 1
    #     for l in 1:cone.d, k in 1:l
    #         Hww[idx1, idx2] += sum(
    #             (w_vecs[k, m] * w_vecs[i, m] * w_vecs[j, n] * w_vecs[l, n] * diff_mat[m, n] / z) *
    #             (m == n ? 1 : 2) * (i == j ? 1 : sqrt(2)) * (k == l ? 1 : sqrt(2))
    #             for n in 1:cone.d for m in 1:n)
    #         idx2 += 1
    #     end
    #     idx1 += 1
    # end

    cone.hess_updated = true
    return cone.hess
end


function correction(cone::EpiPerTraceEntropyTri , primal_dir::AbstractVector)
    @assert cone.grad_updated

    return corr
end
