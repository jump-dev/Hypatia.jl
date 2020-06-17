#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

hypograph of the root determinant of a (row-wise lower triangle) symmetric positive definite matrix
(u in R, W in S_n+) : u <= det(W)^(1/n)

SC barrier from correspondence with A. Nemirovski
-(5 / 3) ^ 2 * (log(det(W) ^ (1 / n) - u) + logdet(W))

TODO
- describe complex case
=#

mutable struct HypoRootdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    side::Int
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}
    rt2::T
    sc_const::T
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    hess_fact_updated::Bool
    scal_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    W::Matrix{R}
    work_mat::Matrix{R}
    work_mat2::Matrix{R}
    fact_W
    Wi::Matrix{R}
    rootdet::T
    rootdetu::T
    sigma::T
    kron_const::T
    dot_const::T

    correction::Vector{T}

    function HypoRootdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        sc_const::Real = T(25) / T(9),
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        if R <: Complex
            side = isqrt(dim - 1) # real lower triangle and imaginary under diagonal
            @assert side^2 == dim - 1
            cone.is_complex = true
        else
            side = round(Int, sqrt(0.25 + 2 * (dim - 1)) - 0.5)
            @assert side * (side + 1) == 2 * (dim - 1)
            cone.is_complex = false
        end
        cone.side = side
        cone.sc_const = sc_const
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

reset_data(cone::HypoRootdetTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.hess_fact_updated = cone.scal_hess_updated = false)

use_nt(::HypoRootdetTri) = false

use_scaling(::HypoRootdetTri) = false

use_correction(::HypoRootdetTri) = true

function setup_data(cone::HypoRootdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.W = zeros(R, cone.side, cone.side)
    # cone.Wi = zeros(R, cone.side, cone.side)
    cone.work_mat = zeros(R, cone.side, cone.side)
    cone.work_mat2 = zeros(R, cone.side, cone.side)
    cone.correction = zeros(T, dim)
    return
end

get_nu(cone::HypoRootdetTri) = (cone.side + 1) * cone.sc_const

function set_initial_point(arr::AbstractVector{T}, cone::HypoRootdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    side = cone.side
    const1 = sqrt(T(5side^2 + 2side + 1))
    const2 = arr[1] = -sqrt(cone.sc_const * (3side + 1 - const1) / T(2side + 2))
    const3 = -const2 * (side + 1 + const1) / 2side
    incr = (cone.is_complex ? 2 : 1)
    k = 2
    @inbounds for i in 1:side
        arr[k] = const3
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoRootdetTri)
    @assert !cone.feas_updated
    u = cone.point[1]

    @views svec_to_smat!(cone.W, cone.point[2:end], cone.rt2)
    cone.fact_W = cholesky!(Hermitian(cone.W, :U), check = false) # mutates W, which isn't used anywhere else
    if isposdef(cone.fact_W)
        cone.rootdet = exp(logdet(cone.fact_W) / cone.side)
        cone.rootdetu = cone.rootdet - u
        cone.is_feas = (cone.rootdetu > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

# update_dual_feas(cone::HypoRootdetTri) = true
function update_dual_feas(cone::HypoRootdetTri)
    u = cone.dual_point[1]
    if u < 0
        @views svec_to_smat!(cone.work_mat, cone.dual_point[2:end], cone.rt2)
        dual_fact_W = cholesky!(Hermitian(cone.work_mat, :U), check = false)
        return isposdef(dual_fact_W) && (logdet(dual_fact_W) > cone.side * log(-u / cone.side))
    else
        return false
    end
end

function update_grad(cone::HypoRootdetTri)
    @assert cone.is_feas
    u = cone.point[1]

    cone.grad[1] = cone.sc_const / cone.rootdetu
    # copyto!(cone.Wi, cone.fact_W.factors)
    # LinearAlgebra.inv!(Cholesky(cone.Wi, 'U', 0)) # TODO inplace for bigfloat
    cone.Wi = inv(cone.fact_W)
    @views smat_to_svec!(cone.grad[2:cone.dim], cone.Wi, cone.rt2)
    cone.sigma = cone.rootdet / cone.rootdetu / cone.side
    @. cone.grad[2:end] *= -cone.sc_const * (cone.sigma + 1)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoRootdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    Wi = cone.Wi
    H = cone.hess.data
    side = cone.side

    idx_incr = (cone.is_complex ? 2 : 1)
    for i in 1:side
        for j in 1:(i - 1)
            row_idx = (cone.is_complex ? (i - 1)^2 + 2j : 1 + div((i - 1) * i, 2) + j)
            col_idx = row_idx
            @inbounds for k in i:side
                @inbounds for l in (i == k ? j : 1):(k - 1)
                    terma = Wi[k, i] * Wi[j, l]
                    termb = Wi[l, i] * Wi[j, k]
                    Wiji = Wi[j, i]
                    Wilk = Wi[l, k]
                    term1 = (terma + termb) * cone.kron_const + Wiji * 2 * real(Wilk) * cone.dot_const
                    H[row_idx, col_idx] = real(term1)
                    @inbounds if cone.is_complex
                        H[row_idx + 1, col_idx] = -imag(term1)
                        term2 = (terma - termb) * cone.kron_const - Wiji * 2im * imag(Wilk) * cone.dot_const
                        H[row_idx, col_idx + 1] = imag(term2)
                        H[row_idx + 1, col_idx + 1] = real(term2)
                    end
                    col_idx += idx_incr
                end

                l = k
                term = cone.rt2 * (Wi[i, k] * Wi[k, j] * cone.kron_const + Wi[i, j] * Wi[k, k] * cone.dot_const)
                H[row_idx, col_idx] = real(term)
                @inbounds if cone.is_complex
                    H[row_idx + 1, col_idx] = imag(term)
                end
                col_idx += 1
            end
        end

        j = i
        row_idx = (cone.is_complex ? (i - 1)^2 + 2j : 1 + div((i - 1) * i, 2) + j)
        col_idx = row_idx
        @inbounds for k in i:side
            @inbounds for l in (i == k ? j : 1):(k - 1)
                term = cone.rt2 * (Wi[k, i] * Wi[j, l] * cone.kron_const + Wi[i, j] * Wi[k, l] * cone.dot_const)
                H[row_idx, col_idx] = real(term)
                @inbounds if cone.is_complex
                    H[row_idx, col_idx + 1] = imag(term)
                end
                col_idx += idx_incr
            end

            l = k
            H[row_idx, col_idx] = abs2(Wi[k, i]) * cone.kron_const + real(Wi[i, i] * Wi[k, k]) * cone.dot_const
            col_idx += 1
        end
    end
    @. H[2:end, 2:end] *= cone.sc_const

    cone.hess_updated = true
    return cone.hess
end

# update first row of the Hessian
function update_hess_prod(cone::HypoRootdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.grad_updated

    sigma = cone.sigma # rootdet / rootdetu / side
    # update constants used in the Hessian
    cone.kron_const = sigma + 1
    cone.dot_const = sigma * (sigma - inv(T(cone.side)))
    # update first row in the Hessian
    hess = cone.hess.data
    @. hess[1, :] = cone.grad / cone.rootdetu
    @. hess[1, 2:end] *= sigma / cone.kron_const

    cone.hess_prod_updated = true
    return
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoRootdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end

    const_diag = cone.dot_const / cone.kron_const
    @views mul!(prod[1, :]', cone.hess[1, :]', arr)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.work_mat, view(arr, 2:cone.dim, i), cone.rt2)
        copytri!(cone.work_mat, 'U', cone.is_complex)
        rdiv!(cone.work_mat, cone.fact_W)
        const_i = tr(cone.work_mat) * const_diag
        for j in 1:cone.side
            @inbounds cone.work_mat[j, j] += const_i
        end
        ldiv!(cone.fact_W, cone.work_mat)
        smat_to_svec!(view(prod, 2:cone.dim, i), cone.work_mat, cone.rt2)
    end
    @views mul!(prod[2:cone.dim, :], cone.hess[2:end, 1], arr[1, :]', true, cone.sc_const * cone.kron_const)

    return prod
end

function third_skron(i, j, k, l, m, n, Wi::Symmetric{T, Matrix{T}}) where {T}
    rt2 = sqrt(T(2))
    t1 = Wi[i, m] * Wi[n, k] * Wi[l, j] + Wi[i, k] * Wi[l, m] * Wi[n, j]
    if m != n
        t1 += Wi[i, n] * Wi[m, k] * Wi[l, j] + Wi[i, k] * Wi[l, n] * Wi[m, j]
        if k != l
            t1 += Wi[i, m] * Wi[n, l] * Wi[k, j] + Wi[i, l] * Wi[k, m] * Wi[n, j] + Wi[i, n] * Wi[m, l] * Wi[k, j] + Wi[i, l] * Wi[k, n] * Wi[m, j]  #################
            if i != j
                t1 += Wi[j, m] * Wi[n, k] * Wi[l, i] + Wi[j, k] * Wi[l, m] * Wi[n, i] +
                    Wi[j, n] * Wi[m, k] * Wi[l, i] + Wi[j, k] * Wi[l, n] * Wi[m, i] +
                    Wi[j, m] * Wi[n, l] * Wi[k, i] + Wi[j, l] * Wi[k, m] * Wi[n, i] + Wi[j, n] * Wi[m, l] * Wi[k, i] + Wi[j, l] * Wi[k, n] * Wi[m, i]
            end
        elseif i != j
            t1 += Wi[j, m] * Wi[n, k] * Wi[l, i] + Wi[j, k] * Wi[l, m] * Wi[n, i] + Wi[j, n] * Wi[m, k] * Wi[l, i] + Wi[j, k] * Wi[l, n] * Wi[m, i]
        end
    elseif k != l
        t1 += Wi[i, m] * Wi[n, l] * Wi[k, j] + Wi[i, l] * Wi[k, m] * Wi[n, j]
        if i != j
            t1 += Wi[j, m] * Wi[n, k] * Wi[l, i] + Wi[j, k] * Wi[l, m] * Wi[n, i] + Wi[j, m] * Wi[n, l] * Wi[k, i] + Wi[j, l] * Wi[k, m] * Wi[n, i]
        end
    elseif i != j
        t1 += Wi[j, m] * Wi[n, k] * Wi[l, i] + Wi[j, k] * Wi[l, m] * Wi[n, i]
    end

    num_match = (i == j) + (k == l) + (m == n)
    if num_match == 0
        t1 /= rt2 * 2
    elseif num_match == 1
        t1 /= 2
    elseif num_match == 2
        t1 /= rt2
    end

    return t1
end
function skron(i, j, k, l, W::Symmetric{T, Matrix{T}}) where {T}
    if (i == j) && (k == l)
        return abs2(W[i, k])
    elseif (i == j) || (k == l)
        return sqrt(T(2)) * W[k, i] * W[j, l]
    else
        return W[k, i] * W[j, l] + W[l, i] * W[j, k]
    end
end

# TODO allocs
function correction2(cone::HypoRootdetTri{T}, primal_dir::AbstractVector{T}, dual_dir::AbstractVector{T}) where {T}
    u_dir = primal_dir[1]
    @views w_dir = primal_dir[2:end]

    side = cone.side
    sigma = cone.sigma
    Wi = Symmetric(cone.Wi)
    dim = cone.dim
    w_dim = dim - 1
    theta = cone.sc_const
    z = cone.rootdetu
    rho(T, i, j) = (i == j ? one(T) : sqrt(T(2)))

    Tuuu = 2 / z / z / z

    # debug against explicit multiplication
    # third = zeros(dim, dim, dim)
    # third_debug = zeros(dim, dim, dim)
    #
    # third[1, 1, 1] = 2 / z ^ 3
    # idx1 = 2
    # for j in 1:side, i in 1:j
    #     # Tuuw
    #     third[1, 1, idx1] = third[1, idx1, 1] = third[idx1, 1, 1] = -2 * sigma / abs2(z) * Wi[j, i] * rho(T, i, j)
    #     idx2 = 2
    #     # Tuww
    #     for l in 1:side, k in 1:l
    #         (idx2 <= idx1) || continue
    #         third[1, idx1, idx2] = third[1, idx2, idx1] = third[idx1, 1, idx2] =
    #             third[idx2, 1, idx1] = third[idx1, idx2, 1] = third[idx2, idx1, 1] =
    #             rho(T, i, j) * rho(T, k, l) * Wi[j, i] * Wi[l, k] * sigma / z * (2 * sigma - inv(side)) + sigma / z * skron(i, j, k, l, Wi)
    #         idx3 = 2
    #         # Twww
    #         for n in 1:side, m in 1:n
    #             (idx3 <= idx2) || continue
    #             # don't forget to negate skron
    #             third[idx1, idx2, idx3] = third[idx1, idx3, idx2] = third[idx2, idx1, idx3] =
    #                 third[idx2, idx3, idx1] = third[idx3, idx1, idx2] = third[idx3, idx2, idx1] =
    #                 -rho(T, i, j) * rho(T, k, l) * rho(T, m, n) * Wi[i, j] * Wi[k, l] * Wi[m, n] * sigma * (inv(side) - sigma) * (inv(side) - 2 * sigma) +
    #                 sigma * (inv(side) - sigma) * (rho(T, i, j) * Wi[i, j] * skron(k, l, m, n, Wi) + rho(T, k, l) * Wi[k, l] * skron(i, j, m, n, Wi) + rho(T, m, n) * Wi[m, n] * skron(i, j, k, l, Wi)) +
    #                 -(sigma + 1) * third_skron(i, j, k, l, m, n, Wi)
    #             third_debug[idx1, idx2, idx3] = third_debug[idx1, idx3, idx2] = third_debug[idx2, idx1, idx3] =
    #                 third_debug[idx2, idx3, idx1] = third_debug[idx3, idx1, idx2] = third_debug[idx3, idx2, idx1] =
    #                 # -(sigma + 1) * third_skron(i, j, k, l, m, n, Wi)
    #                 sigma * (inv(side) - sigma) * (rho(T, i, j) * Wi[i, j] * skron(k, l, m, n, Wi) + rho(T, k, l) * Wi[k, l] * skron(i, j, m, n, Wi) + rho(T, m, n) * Wi[m, n] * skron(i, j, k, l, Wi))
    #                 # -rho(T, i, j) * rho(T, k, l) * rho(T, m, n) * Wi[i, j] * Wi[k, l] * Wi[m, n] * sigma * (inv(side) - sigma) * (inv(side) - 2 * sigma)
    #             idx3 += 1
    #         end
    #         idx2 += 1
    #     end
    #     idx1 += 1
    # end
    # third *= theta
    # third_debug *= theta

    # first term, 3rd order kron
    S = copytri!(svec_to_smat!(similar(cone.work_mat), w_dir, cone.rt2), 'U', cone.is_complex) # TODO allocates
    ldiv!(cone.fact_W, S)
    rdiv!(S, cone.fact_W.U)
    mul!(cone.work_mat2, S, S') # TODO use outer prod function
    term1 = -2 * theta * (sigma + 1) * smat_to_svec!(zeros(T, cone.dim - 1), cone.work_mat2, cone.rt2)
    # @show term1 ./ (reshape(reshape(third_debug[2:end, 2:end, 2:end], w_dim^2, w_dim) * w_dir, w_dim, w_dim) * w_dir)


    # second term, 2nd order kron and a dot
    S = copytri!(svec_to_smat!(cone.work_mat, w_dir, cone.rt2), 'U', cone.is_complex)
    # keep a kron product handy, could factor out
    skron2 = Wi * Symmetric(S) * Wi # TODO ldiv rdiv
    dot_skron = dot(skron2, S)
    dot_Wi_S = dot(Wi, S)
    term2 = zeros(T, w_dim)
    idx = 1
    for j in 1:side, i in 1:j
        term2[idx] = rho(T, i, j) * (2 * skron2[i, j] * dot_Wi_S + Wi[i, j] * dot_skron)
        idx += 1
    end
    scal1 = theta * sigma * (inv(T(side)) - sigma)
    term2 .*= scal1
    # @show term2 ./ (reshape(reshape(third_debug[2:end, 2:end, 2:end], w_dim^2, w_dim) * w_dir, w_dim, w_dim) * w_dir)
    #
    # third term, purely dots
    scal2 = dot_Wi_S * (inv(T(side)) - sigma - sigma)
    scal3 = -scal1 * scal2 * dot_Wi_S
    term3 = scal3 * [Wi[i, j] * rho(T, i, j) for j in 1:side for i in 1:j]
    # @show term3 ./ (reshape(reshape(third_debug[2:end, 2:end, 2:end], w_dim^2, w_dim) * w_dir, w_dim, w_dim) * w_dir)
    #
    # now take into account hypograph variable
    # corr[k] += 2 * sum(third[1, j, k] * u_dir * primal_dir[j] for j in 2:dim)
    # add on contributions from Tuww
    # S = copytri!(svec_to_smat!(similar(cone.work_mat), w_dir, cone.rt2), 'U', cone.is_complex) # TODO allocates
    term4a = smat_to_svec!(zeros(T, w_dim), skron2, cone.rt2)
    term4b = -scal2 * smat_to_svec!(zeros(T, w_dim), Wi, cone.rt2)
    term4ab = sigma / z * (term4a + term4b)
    term4 = 2 * u_dir * theta * term4ab

    term5a = -2 * sigma / abs2(z) * smat_to_svec!(zeros(T, w_dim), Wi, cone.rt2)
    term5 = term5a * u_dir * theta * u_dir


    corr = similar(cone.correction)
    corr[2:dim] = term1 + term2 + term3 + term4 + term5
    # for k in 2:dim
        # corr[k] += third[1, 1, k] * abs2(u_dir)
        # corr[k] += 2 * u_dir * sum(third[1, j, k] * primal_dir[j] for j in 2:dim)
    # end
    # corr[1] = sum(third[1, j, k] * primal_dir[j] * primal_dir[k] for j in 1:dim for k in 1:dim)
    corr[1] = theta * (
        dot(term4ab, primal_dir[2:dim]) +
        # sum(third[1, j, k] * primal_dir[j] * primal_dir[k] for j in 2:dim for k in 2:dim) +
        # 2 * u_dir * sum(third[1, 1, j] * primal_dir[j] for j in 2:dim) +
        u_dir * (2 * dot(term5a, primal_dir[2:dim]) +
        Tuuu * u_dir)
        )
    # @show corr ./ (reshape(reshape(third, dim^2, dim) * primal_dir, dim, dim) * primal_dir / -2)
    cone.correction = corr / -2


    # cone.correction = reshape(reshape(third, dim^2, dim) * primal_dir, dim, dim) * primal_dir / -2
    return cone.correction
end
