#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) epigraph of sum of perspectives of entropies (AKA vector relative entropy cone)
(u in R, v in R_+^n, w in R_+^n) : u >= sum_i w_i*log(w_i/v_i) TODO update description here for non-contiguous v/w

barrier from "Primal-Dual Interior-Point Methods for Domain-Driven Formulations" by Karimi & Tuncel, 2019
-log(u - sum_i w_i*log(w_i/v_i)) - sum_i (log(v_i) + log(w_i))

TODO
- write native tests for use_dual = true
- update examples for non-contiguous v/w
- keep continguous copies?
=#

mutable struct EpiSumPerEntropy{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    w_dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, SparseMatrixCSC{T, Int}}
    hess_fact_cache
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    v_idxs
    w_idxs
    tau::Vector{T}
    z::T
    sigma::Vector{T}
    Hu::Vector{T} # first row of inverse Hessian
    Huu::T # first element of inverse Hessian
    Hvv::Vector{T} # part of inverse Hessian 0th diagonal
    Hvw::Vector{T} # part of inverse Hessian 1st diagonal
    Hww::Vector{T} # part of inverse Hessian 0th diagonal
    denom::Vector{T} # denominator for all parts but the first of the inverse Hessian
    detdiag::Vector{T} # determinant of blocks on the diagonal of the inverse Hessian

    function EpiSumPerEntropy{T}(
        dim::Int;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
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
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

reset_data(cone::EpiSumPerEntropy) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.hess_fact_updated = cone.inv_hess_updated = cone.hess_inv_hess_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiSumPerEntropy{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    w_dim = cone.w_dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.tau = zeros(T, w_dim)
    cone.sigma = zeros(T, w_dim)
    cone.Hu = zeros(T, dim - 1)
    cone.Hvv = zeros(T, w_dim)
    cone.Hvw = zeros(T, w_dim)
    cone.Hww = zeros(T, w_dim)
    cone.denom = zeros(T, w_dim)
    cone.detdiag = zeros(T, w_dim)
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
        @. cone.tau = log(w / v)
        cone.z = u - dot(w, cone.tau)
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
        # TODO allocates
        return all(u * (1 .+ log.(v / u)) + w .> 0)
        # return all(v .> u .* exp.(-w ./ u .- 1))
    else
        return false
    end
end

function update_grad(cone::EpiSumPerEntropy)
    @assert cone.is_feas
    u = cone.point[1]
    @views v = cone.point[cone.v_idxs]
    @views w = cone.point[cone.w_idxs]
    z = cone.z
    g = cone.grad
    tau = cone.tau

    @. tau += 1
    @. tau /= -z
    g[1] = -inv(z)
    @. g[cone.v_idxs] = (-w / z - 1) / v
    @. g[cone.w_idxs] = -inv(w) - tau

    cone.grad_updated = true
    return cone.grad
end

function update_hess_inv_hess(cone::EpiSumPerEntropy)
    @assert !cone.hess_inv_hess_updated
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]
    tau = cone.tau
    z = cone.z
    sigma = cone.sigma
    H = cone.hess.data

    # @inbounds for (j, wj) in enumerate(w)
    #     wdenj = cone.wden[j]
    #     invdenj = 2 / cone.den[j]
    #
    #         d11 = cone.diag[2j - 1] = abs2(real(wdenj)) + invdenj
    #         d22 = cone.diag[2j] = abs2(imag(wdenj)) + invdenj
    #         d12 = cone.offdiag[j] = real(wdenj) * imag(wdenj)
    #         cone.detdiag[j] = d11 * d22 - abs2(d12)
    #
    #     u2pwj2 = usqr + abs2(wj)
    #     invedge[j] = 2 * u / u2pwj2 * wj
    #     schur += 2 / u2pwj2
    # end

    # H_u_u, H_u_v, H_u_w parts
    H[1, 1] = abs2(cone.grad[1])
    @. sigma = w / v / z
    @. H[1, v_idxs] = sigma / z
    @. H[1, w_idxs] = tau / z

    cone.hess_inv_hess_updated = true
    return
end

function update_hess(cone::EpiSumPerEntropy)
    @assert cone.grad_updated
    cone.hess_inv_hess_updated || update_hess_inv_hess(cone) # H_u_u, H_u_v, H_u_w parts
    w_dim = cone.w_dim
    u = cone.point[1]
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]
    tau = cone.tau
    z = cone.z
    sigma = cone.sigma
    H = cone.hess.data

    # H_v_v, H_v_w, H_w_w parts
    @inbounds for (i, v_idx, w_idx) in zip(1:w_dim, v_idxs, w_idxs)
        vi = point[v_idx]
        wi = point[w_idx]
        taui = tau[i]
        sigmai = sigma[i]
        invvi = inv(vi)

        H[v_idx, v_idx] = abs2(sigmai) + (sigmai + invvi) / vi
        H[w_idx, w_idx] = abs2(taui) + (inv(z) + inv(wi)) / wi

        @. H[v_idx, w_idxs] = sigmai * tau
        @. H[w_idx, v_idxs] = sigma * taui
        H[v_idx, w_idx] -= invvi / z

        @inbounds for j in (i + 1):w_dim
            H[v_idx, v_idxs[j]] = sigmai * sigma[j]
            H[w_idx, w_idxs[j]] = taui * tau[j]
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# auxiliary calculations for inverse Hessian and inverse Hessian prod
function inv_hess_vals(cone::EpiSumPerEntropy{T}) where {T}
    cone.hess_inv_hess_updated || update_hess_inv_hess(cone)
    u = cone.point[1]
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]
    z = cone.z
    Hu = cone.Hu
    logprod = u - z # TODO cache in feas check?
    @. cone.denom = z + 2 * w

    for (i, v_idx, w_idx) in zip(1:cone.w_dim, v_idxs, w_idxs)
        temp1 = logprod - w[i] * log(w[i] / v[i]) # TODO cache in feas check?
        temp2 = log(w[i] / v[i]) # TODO cache in feas check?
        Hu[v_idx - 1] = -(u - temp1 - 2 * w[i] * temp2) * w[i] * v[i] / cone.denom[i]
        Hu[w_idx - 1] = abs2(w[i]) * (temp2 * z + u - temp1) / cone.denom[i]
    end
    @views cone.Huu = abs2(z) * (1 - dot(Hu, cone.hess[1, 2:cone.dim]))
    @. cone.Hvw = v * abs2(w)
    @. cone.Hvv = z + w
    @. cone.Hww = cone.Hvv * abs2(w)
    @. cone.Hvv *= abs2(v)
    @. cone.detdiag = (cone.Hvv * cone.Hww - abs2(cone.Hvw)) / cone.denom / cone.denom # TODO this isn't used outside of inv_hess_sqrt_prod so just move it there and don't cache

    return
end

function update_inv_hess(cone::EpiSumPerEntropy{T}) where {T}
    inv_hess_vals(cone)
    dim = cone.dim
    w_dim = cone.w_dim
    if !isdefined(cone, :inv_hess)
        # initialize sparse idxs for upper triangle of Hessian
        dim = cone.dim
        H_nnz_tri = 2 * dim - 1 + w_dim
        I = Vector{Int}(undef, H_nnz_tri)
        J = Vector{Int}(undef, H_nnz_tri)
        idxs1 = 1:dim
        I[idxs1] .= 1
        J[idxs1] .= idxs1
        idxs2 = (dim + 1):(2 * dim - 1)
        I[idxs2] .= 2:dim
        J[idxs2] .= 2:dim
        idxs3 = (2 * dim):H_nnz_tri
        I[idxs3] .= 2:2:dim
        J[idxs3] .= 3:2:dim
        V = ones(T, H_nnz_tri)
        cone.inv_hess = Symmetric(sparse(I, J, V, dim, dim), :U)
    end

    # modify nonzeros of sparse data structure of upper triangle of Hessian
    H_nzval = cone.inv_hess.data.nzval
    H_nzval[1] = cone.Huu
    vw_idx = 1
    nz_idx = 2
    dim_idx = 1
    @inbounds for j in 1:w_dim
        H_nzval[nz_idx] = cone.Hu[dim_idx]
        H_nzval[nz_idx + 1] = cone.Hvv[vw_idx] / cone.denom[vw_idx]
        nz_idx += 2
        dim_idx += 1
        H_nzval[nz_idx] = cone.Hu[dim_idx]
        H_nzval[nz_idx + 1] = cone.Hvw[vw_idx] / cone.denom[vw_idx]
        # @show typeof(cone.Hvw)
        H_nzval[nz_idx + 2] = cone.Hww[vw_idx] / cone.denom[vw_idx]
        nz_idx += 3
        vw_idx += 1
        dim_idx += 1
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiSumPerEntropy{T}) where {T <: Real}
    # updates for nonzero values in the inverse Hessian
    inv_hess_vals(cone) # TODO only do once
    Hu = cone.Hu
    Huu = cone.Huu
    Hvv = cone.Hvv
    Hvw = cone.Hvw
    Hww = cone.Hww
    denom = cone.denom

    @. @views prod[1, :] = arr[1, :] * Huu
    @views mul!(prod[1, :], arr[2:end, :]', Hu, true, true)
    @. @views prod[2:end, :] = Hu * arr[1, :]'
    @inbounds for i in 1:cone.w_dim
        @. @views prod[2i, :] += (Hvv[i] * arr[2i, :] + Hvw[i] * arr[2i + 1, :]) / denom[i]
        @. @views prod[2i + 1, :] += (Hww[i] * arr[2i + 1, :] + Hvw[i] * arr[2i, :]) / denom[i]
    end

    return prod
end

# function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiSumPerEntropy)
#     if !cone.hess_inv_hess_updated
#         update_hess_inv_hess(cone) # TODO needed?
#     end
#
#     u = cone.point[1]
#     v_idxs = cone.v_idxs
#     w_idxs = cone.w_idxs
#     point = cone.point
#     @views v = point[v_idxs]
#     @views w = point[w_idxs]
#
#     # schur = cone.Huu
#     # for i in 1:cone.w_dim
#     #     a = [cone.Hu[2i - 1], cone.Hu[2i]]
#     #     # Hu is already divided by denom, diagonal blocks are not
#     #     A = Symmetric([cone.Hvv[i] cone.Hvw[i]; cone.Hvw[i] cone.Hww[i]] ./ cone.denom[i])
#     #     schur -= dot(a, A \ a)
#     # end
#     # @show Matrix(cone.inv_hess)
#     # @show schur
#     # @show cone.denom
#     # A = Matrix(cone.inv_hess)
#     # schur = cone.Huu - dot(A[2:end, 1], A[2:end, 2:end] * A[2:end, 1])
#     # @show dot(A[2:end, 1], A[2:end, 2:end] * A[2:end, 1])
#     schur = cone.Huu
#     @. @views prod[1, :] = sqrt(schur) * arr[1, :]
#
#     for (j, oj) in enumerate(cone.Hvw)
#         # TODO cache these fields?
#         evj = cone.Hu[2j - 1]
#         ewj = cone.Hu[2j]
#         # rtd1j = sqrt(cone.diag[2j - 1])
#         rtd1j = sqrt(cone.Hvv[j] / cone.denom[j])
#         rtdetj = sqrt(cone.detdiag[j])
#         ortd1j = oj / rtd1j
#         side1j = evj / rtd1j
#         side2j = (ewj * rtd1j - evj * ortd1j) / rtdetj
#         rtdetd1j = rtdetj / rtd1j
#         # @show side1j, side2j, rtdetd1j, ortd1j, rtd1j
#         @. @views prod[2j, :] = side1j * arr[1, :] + rtd1j * arr[2j, :] + ortd1j * arr[2j + 1, :]
#         @. @views prod[2j + 1, :] = side2j * arr[1, :] + rtdetd1j * arr[2j + 1, :]
#
#         # @. @views prod[2j, :] *= sqrt(cone.denom[j])
#         # @. @views prod[2j + 1, :] *= sqrt(cone.denom[j])
#     end
#     # @show prod
#     @show prod ./ cholesky(Matrix(cone.inv_hess)).L
#
#     return prod
# end

function correction2(cone::EpiSumPerEntropy{T}, primal_dir::AbstractVector{T}) where {T <: Real}
    @assert cone.hess_updated
    tau = cone.tau
    sigma = cone.sigma
    z = cone.z
    w_dim = cone.w_dim
    u = cone.point[1]
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]

    u_dir = primal_dir[1]
    v_dir = primal_dir[v_idxs]
    w_dir = primal_dir[w_idxs]

    third = zeros(T, cone.dim, cone.dim, cone.dim)
    third_debug = zeros(T, cone.dim, cone.dim, cone.dim)
    dim = cone.dim
    corr = cone.correction
    v_corr = view(corr, v_idxs)
    w_corr = view(corr, w_idxs)
    corr .= 0

    # Tuuu
    third[1, 1, 1] = -2 / z ^ 3
    third_debug[1, 1, 1] = -2 / z ^ 3

    Tuuu = -2 / z ^ 3
    corr[1] = Tuuu * abs2(u_dir)

    scal_uvv = -2 / z / z * u_dir

    # Tuuv
    third[1, 1, v_idxs] = third[1, v_idxs, 1] = third[v_idxs, 1, 1] = -2 * sigma / abs2(z)
    third_debug[1, 1, v_idxs] = third_debug[1, v_idxs, 1] = third_debug[v_idxs, 1, 1] = -2 * sigma / abs2(z)
    corr[1] += scal_uvv * dot(sigma, v_dir) * 2
    v_corr .+= scal_uvv * u_dir .* sigma

    # Tuuw
    third[1, 1, w_idxs] = third[1, w_idxs, 1] = third[w_idxs, 1, 1] = -2 * tau / abs2(z)
    third_debug[1, 1, w_idxs] = third_debug[1, w_idxs, 1] = third_debug[w_idxs, 1, 1] = -2 * tau / abs2(z)
    corr[1] += scal_uvv * dot(tau, w_dir) * 2
    w_corr .+= scal_uvv * u_dir .* tau # note (scal_uvv * u_dir) already calculated in uuv


    # Tuvv
    third[1, v_idxs, v_idxs] = third[v_idxs, 1, v_idxs] = third[v_idxs, v_idxs, 1] = -2 * sigma * sigma' / z - Diagonal(sigma ./ v / z)
    third_debug[1, v_idxs, v_idxs] = third_debug[v_idxs, 1, v_idxs] = third_debug[v_idxs, v_idxs, 1] = -2 * sigma * sigma' / z - Diagonal(sigma ./ v / z)
    scal_uvv_ij = -2 / z
    corr[1] += scal_uvv_ij * abs2(dot(v_dir, sigma)) - dot(sigma ./ v, abs2.(v_dir)) / z
    v_corr .+= 2 * u_dir * sigma .* (scal_uvv_ij .* dot(v_dir, sigma) .- v_dir ./ v / z)

    # Tuvw
    third[1, v_idxs, w_idxs] = third[v_idxs, 1, w_idxs] = third[v_idxs, w_idxs, 1] =
        -2 * sigma * tau' / z + Diagonal(inv.(v) / abs2(z))
    third[1, w_idxs, v_idxs] = third[w_idxs, 1, v_idxs] = third[w_idxs, v_idxs, 1] =
        -2 * tau * sigma' / z + Diagonal(inv.(v) / abs2(z))
    third_debug[1, v_idxs, w_idxs] = third_debug[v_idxs, 1, w_idxs] = third_debug[v_idxs, w_idxs, 1] =
        -2 * sigma * tau' / z + Diagonal(inv.(v) / abs2(z))
    third_debug[1, w_idxs, v_idxs] = third_debug[w_idxs, 1, v_idxs] = third_debug[w_idxs, v_idxs, 1] =
        -2 * tau * sigma' / z + Diagonal(inv.(v) / abs2(z))
    scal_uvw = -2 / z
    corr[1] += 2 * (scal_uvw * dot(tau, w_dir) * dot(sigma, v_dir) + dot(inv.(v) / abs2(z), v_dir .* w_dir))
    v_corr .+= 2 * u_dir * (scal_uvw * dot(tau, w_dir) * sigma .+ w_dir ./ v ./ z ./ z)
    w_corr .+= 2 * u_dir * (scal_uvw * dot(sigma, v_dir) * tau .+ v_dir ./ v ./ z ./ z)

    # Tuww
    third[1, w_idxs, w_idxs] = third[w_idxs, 1, w_idxs] = third[w_idxs, w_idxs, 1] = -2 * tau * tau' / z - Diagonal(inv.(w) / abs2(z))
    third_debug[1, w_idxs, w_idxs] = third_debug[w_idxs, 1, w_idxs] = third_debug[w_idxs, w_idxs, 1] = -2 * tau * tau' / z - Diagonal(inv.(w) / abs2(z))
    scal_uww_ij = -2 / z
    corr[1] += scal_uww_ij * abs2(dot(tau, w_dir)) - dot(inv.(w) / z / z, abs2.(w_dir))
    w_corr .+= 2 * u_dir * (scal_uww_ij .* tau .* dot(tau, w_dir) .- w_dir ./ w / z / z)

    # Tvvv
    for i in 1:w_dim, j in 1:w_dim, k in 1:w_dim
        (ti, tj, tk) = (v_idxs[i], v_idxs[j], v_idxs[k])
        t1 = -2 * sigma[i] * sigma[j] * sigma[k]
        if i == j
            t2 = -sigma[i] * sigma[k] / v[i]
            if j == k
                third[ti, ti, ti] = t1 + 3 * t2 - 2 * sigma[i] / abs2(v[i]) - 2 / v[i] ^ 3
                # third_debug[ti, ti, ti] = t1 + 3 * t2 - 2 * sigma[i] / abs2(v[i]) - 2 / v[i] ^ 3
                third_debug[ti, ti, ti] = t1 + 3 * t2 - 2 * sigma[i] / abs2(v[i]) - 2 / v[i] ^ 3
            else
                third[ti, ti, tk] = third[ti, tk, ti] = third[tk, ti, ti] = t1 + t2
                third_debug[ti, ti, tk] = third_debug[ti, tk, ti] = third_debug[tk, ti, ti] = t1 + t2
            end
        elseif i != k && j != k
            third[ti, tj, tk] = third[ti, tk, tj] = third[tj, ti, tk] = third[tj, tk, ti] =
                third[tk, ti, tj] = third[tk, tj, ti] = t1
            third_debug[ti, tj, tk] = third_debug[ti, tk, tj] = third_debug[tj, ti, tk] = third_debug[tj, tk, ti] =
                third_debug[tk, ti, tj] = third_debug[tk, tj, ti] = t1
        end
    end
    v_corr .+=
        -2 * abs2(dot(sigma, v_dir)) .* sigma +
        -sigma .* (dot(sigma ./ v, abs2.(v_dir)) .+ 2 * dot(sigma, v_dir) .* v_dir ./ v) +
        -2 * (sigma ./ v ./ v + 1 ./ v .^ 3) .* abs2.(v_dir)


    # Tvvw
    for i in 1:w_dim, j in 1:w_dim, k in 1:w_dim
        # note that the offset for w is not the same as the offset for the vs, so even if i = k, ti != tk, so we make sure to account for that
        (ti, tj, tk) = (v_idxs[i], v_idxs[j], w_idxs[k])
        t1 = -2 * sigma[i] * sigma[j] * tau[k]
        if i == j
            t2 = -sigma[i] * tau[k] / v[i]
            if j == k
                # vi vi wi
                third[ti, ti, tk] = third[ti, tk, ti] = third[tk, ti, ti] = t1 + t2 + 2 * sigma[i] / v[i] / z + inv(abs2(v[i])) / z
                # third_debug[ti, ti, tk] = third_debug[ti, tk, ti] = third_debug[tk, ti, ti] = t1 + t2 + 2 * sigma[i] / v[i] / z + inv(abs2(v[i])) / z
            else
                # vi vi wk
                third[ti, ti, tk] = third[ti, tk, ti] = third[tk, ti, ti] = t1 + t2
                # third_debug[ti, ti, tk] = third_debug[ti, tk, ti] = third_debug[tk, ti, ti] = t1 + t2
            end
        elseif i == k
            # vi vj wj, symmetric to vi vj wi
            third[ti, tj, tk] = third[ti, tk, tj] = third[tj, ti, tk] = third[tj, tk, ti] =
                third[tk, ti, tj] = third[tk, tj, ti] = t1 + sigma[j] / v[i] / z
            # third_debug[ti, tj, tk] = third_debug[ti, tk, tj] = third_debug[tj, ti, tk] = third_debug[tj, tk, ti] =
            #     third_debug[tk, ti, tj] = third_debug[tk, tj, ti] = t1 + sigma[j] / v[i] / z
        elseif j != k
            # vi vj wk
            third[ti, tj, tk] = third[ti, tk, tj] = third[tj, ti, tk] = third[tj, tk, ti] =
                third[tk, ti, tj] = third[tk, tj, ti] = t1
            # third_debug[ti, tj, tk] = third_debug[ti, tk, tj] = third_debug[tj, ti, tk] = third_debug[tj, tk, ti] =
            #     third_debug[tk, ti, tj] = third_debug[tk, tj, ti] = t1
        end
    end
    # v_corr .+= -4 * dot(sigma, v_dir) * dot(tau, w_dir) .* sigma +
    #     -2 * dot(tau, w_dir) .* sigma ./ v .* v_dir +
    #     1 / z * (sigma .* v_dir * dot(w_dir, inv.(v)) + w_dir ./ v * dot(sigma, v_dir)) + # WRONG
    #     inv.(abs2.(v)) .* v_dir .* w_dir / z
    # w_corr .+= -2 * abs2(dot(sigma, v_dir)) .* tau +
    #     -dot(sigma ./ v, abs2.(v_dir)) .* tau +
    #     2 / z * (dot(sigma, v_dir) ) .* v_dir ./ v +
    #     inv.(abs2.(v)) .* abs2.(v_dir) / z

    # Tvww
    for i in 1:w_dim, j in 1:w_dim, k in 1:w_dim
        # note that the offset for v is not the same as the offset for the ws, so even if i = k, ti != tk, so we make sure to account for that
        (ti, tj, tk) = (v_idxs[i], w_idxs[j], w_idxs[k])
        t1 = -2 * sigma[i] * tau[j] * tau[k]
        if j == k
            if i == j
                # vi wi wi
                third[ti, tj, tj] = third[tj, ti, tj] = third[tj, tj, ti] = t1 + 2 * tau[i] / z / v[i] - 1 / v[i] / abs2(z)
                third_debug[ti, tj, tj] = third_debug[tj, ti, tj] = third_debug[tj, tj, ti] = t1 + 2 * tau[i] / z / v[i] - 1 / v[i] / abs2(z)
            else
                # vi wj wj
                third[ti, tj, tj] = third[tj, ti, tj] = third[tj, tj, ti] = t1 - sigma[i] / w[j] / z
                third_debug[ti, tj, tj] = third_debug[tj, ti, tj] = third_debug[tj, tj, ti] = t1 - sigma[i] / w[j] / z
            end
        elseif i == j
            # vi wi wk, symmetric to vi wj wi
            third[ti, tj, tk] = third[ti, tk, tj] = third[tj, ti, tk] = third[tj, tk, ti] =
                third[tk, ti, tj] = third[tk, tj, ti] = t1 + tau[k] / z / v[i]
            third_debug[ti, tj, tk] = third_debug[ti, tk, tj] = third_debug[tj, ti, tk] = third_debug[tj, tk, ti] =
                third_debug[tk, ti, tj] = third_debug[tk, tj, ti] = t1 + tau[k] / z / v[i]
        elseif i != k
            # vi wj wk
            third[ti, tj, tk] = third[ti, tk, tj] = third[tj, ti, tk] = third[tj, tk, ti] =
                third[tk, ti, tj] = third[tk, tj, ti] = t1
            third_debug[ti, tj, tk] = third_debug[ti, tk, tj] = third_debug[tj, ti, tk] = third_debug[tj, tk, ti] =
                third_debug[tk, ti, tj] = third_debug[tk, tj, ti] = t1
        end
    end
    v_corr .+= -2 * abs2(dot(tau, w_dir)) .* sigma +
        -dot(inv.(w), abs2.(w_dir)) .* sigma / z + abs2.(w_dir) ./ w .* sigma / z +
        2 / z * dot(tau, w_dir) .* w_dir ./ v +
        -inv.(v) .* abs2.(w_dir) / z / z
        # inv.(abs2.(v)) .* v_dir .* w_dir / z
    w_corr .+= -4 * dot(tau, w_dir) * dot(sigma, v_dir) .* tau +
        -2 * dot(sigma, v_dir) * w_dir ./ w / z + sigma .* v_dir .* w_dir ./ w / z +
        dot(tau, w_dir) * v_dir ./ v / z + dot(inv.(v), v_dir) * tau .* w_dir / z +  # WRONG
        -inv.(v) .* v_dir .* w_dir / z / z
        # inv.(abs2.(v)) .* abs2.(v_dir) / z
    @show corr ./ (reshape(reshape(third_debug, dim^2, dim) * primal_dir, dim, dim) * primal_dir)

    # Twww
    for i in 1:w_dim, j in 1:w_dim, k in 1:w_dim
        (ti, tj, tk) = (w_idxs[i], w_idxs[j], w_idxs[k])
        t1 = -2 * tau[i] * tau[j] * tau[k]
        if i == j
            t2 = -tau[k] / w[i] / z
            if j == k
                third[ti, ti, ti] = t1 + 3 * t2 - abs2(inv(w[i])) / z - 2 / w[i] ^ 3
                third_debug[ti, ti, ti] = t1 + 3 * t2 - 2 / w[i] ^ 3
            else
                third[ti, ti, tk] = third[ti, tk, ti] = third[tk, ti, ti] = t1 + t2
                third_debug[ti, ti, tk] = third_debug[ti, tk, ti] = third_debug[tk, ti, ti] = t1 + t2
            end
        elseif i != k && j != k
            third[ti, tj, tk] = third[ti, tk, tj] = third[tj, ti, tk] = third[tj, tk, ti] =
                third[tk, ti, tj] = third[tk, tj, ti] = t1
            third_debug[ti, tj, tk] = third_debug[ti, tk, tj] = third_debug[tj, ti, tk] = third_debug[tj, tk, ti] =
                third_debug[tk, ti, tj] = third_debug[tk, tj, ti] = t1
        end
    end
    w_corr .+=
        -2 * abs2(dot(tau, w_dir)) .* tau +
        -(dot(inv.(w), abs2.(w_dir)) .* tau .+ 2 * dot(tau, w_dir) .* w_dir ./ w) / z +
        -2 * (1 ./ w .^ 3) .* abs2.(w_dir)

    third_order = reshape(third, cone.dim^2, cone.dim)

    # barrier = cone.barrier
    # FD_3deriv = ForwardDiff.jacobian(x -> ForwardDiff.hessian(barrier, x), cone.point)
    # @show norm(third_order - FD_3deriv)

    cone.correction .= reshape(third_order * primal_dir, cone.dim, cone.dim) * primal_dir / -2

    return cone.correction
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
