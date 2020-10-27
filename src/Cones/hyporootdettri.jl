#=
hypograph of the root determinant of a (row-wise lower triangle) symmetric positive definite matrix
(u in R, W in S_n+) : u <= det(W)^(1/n)

SC barrier from correspondence with A. Nemirovski
-(5 / 3) ^ 2 * (log(det(W) ^ (1 / n) - u) + logdet(W))

TODO
- describe complex case
=#

mutable struct HypoRootdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    side::Int
    is_complex::Bool
    rt2::T
    sc_const::T

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
    hess_aux_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    W::Matrix{R}
    tmpW::Matrix{R}
    work_mat::Matrix{R}
    work_mat2::Matrix{R}
    fact_W
    Wi::Matrix{R}
    Wi_vec::Vector{T}
    rootdet::T
    rootdetu::T
    sigma::T
    kron_const::T
    dot_const::T
    tmpw::Vector{T}

    function HypoRootdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        sc_const::Real = 25 / T(9),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
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

use_heuristic_neighborhood(cone::HypoRootdetTri) = false

reset_data(cone::HypoRootdetTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated = cone.hess_fact_updated = false)

function setup_extra_data(cone::HypoRootdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    side = cone.side
    cone.W = zeros(R, side, side)
    cone.tmpW = zeros(R, side, side)
    cone.Wi_vec = zeros(T, dim - 1)
    cone.work_mat = zeros(R, side, side)
    cone.work_mat2 = zeros(R, side, side)
    cone.tmpw = zeros(T, dim - 1)
    return cone
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

function update_feas(cone::HypoRootdetTri{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]

    @views svec_to_smat!(cone.W, cone.point[2:end], cone.rt2)
    copyto!(cone.tmpW, cone.W)
    cone.fact_W = cholesky!(Hermitian(cone.tmpW, :U), check = false) # mutates W, which isn't used anywhere else
    if isposdef(cone.fact_W)
        cone.rootdet = exp(logdet(cone.fact_W) / cone.side)
        cone.rootdetu = cone.rootdet - u
        cone.is_feas = (cone.rootdetu > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoRootdetTri{T}) where {T}
    u = cone.dual_point[1]

    if u < -eps(T)
        @views svec_to_smat!(cone.work_mat, cone.dual_point[2:end], cone.rt2)
        dual_fact_W = cholesky!(Hermitian(cone.work_mat, :U), check = false)
        return isposdef(dual_fact_W) && (logdet(dual_fact_W) - cone.side * log(-u / cone.side) > eps(T))
    end

    return false
end

function update_grad(cone::HypoRootdetTri)
    @assert cone.feas_updated
    @assert cone.is_feas
    u = cone.point[1]

    cone.grad[1] = cone.sc_const / cone.rootdetu
    # TODO in-place
    # copyto!(cone.Wi, cone.fact_W.factors)
    # LinearAlgebra.inv!(Cholesky(cone.Wi, 'U', 0)) # TODO inplace for bigfloat
    cone.Wi = inv(cone.fact_W)
    smat_to_svec!(cone.Wi_vec, cone.Wi, cone.rt2)
    cone.sigma = cone.rootdet / cone.rootdetu / cone.side
    @. @views cone.grad[2:end] = -cone.Wi_vec * cone.sc_const * (cone.sigma + 1)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoRootdetTri)
    if !cone.hess_aux_updated
        update_hess_aux(cone)
    end
    Wi = cone.Wi
    H = cone.hess.data
    side = cone.side
    Wi_vec = cone.Wi_vec
    @views Hww = H[2:end, 2:end]

    symm_kron(Hww, Wi, cone.rt2)
    Hww .*= cone.kron_const
    mul!(Hww, Wi_vec, Wi_vec', cone.dot_const, true)
    Hww .*= cone.sc_const

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::HypoRootdetTri)
    H = cone.inv_hess.data
    rootdet = cone.rootdet
    rootdetu = cone.rootdetu
    side = cone.side
    W = Hermitian(cone.W, :U)
    @views w = cone.point[2:end]
    @views Hww = H[2:end, 2:end]

    H[1, 1] = abs2(rootdetu) + abs2(rootdet) / side
    symm_kron(Hww, W, cone.rt2)
    Hww .*= rootdetu * abs2(side)
    mul!(Hww, w, w', rootdet, true)
    denom = side * (side * rootdetu + rootdet) # abs2(side) * rootdetu + side * rootdet # TODO could write using cone.u, check if this is better because rootdetu is derived from u
    Hww ./= denom

    @views H[1, 2:end] = rootdet * w / side
    H ./= cone.sc_const

    cone.inv_hess_updated = true
    return cone.inv_hess
end

# update first row of the Hessian
function update_hess_aux(cone::HypoRootdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.grad_updated

    sigma = cone.sigma # rootdet / rootdetu / side
    # update constants used in the Hessian
    cone.kron_const = sigma + 1
    cone.dot_const = sigma * (sigma - inv(T(cone.side)))
    # update first row in the Hessian
    hess = cone.hess.data
    @. @views hess[1, :] = cone.grad / cone.rootdetu
    @. @views hess[1, 2:end] *= sigma / cone.kron_const

    cone.hess_aux_updated = true
    return
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoRootdetTri)
    if !cone.hess_aux_updated
        update_hess_aux(cone)
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

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoRootdetTri)
    H = cone.inv_hess.data
    rootdet = cone.rootdet
    rootdetu = cone.rootdetu
    side = cone.side
    W = Hermitian(cone.W, :U)
    @views w = cone.point[2:end]
    const1 = rootdet / side

    denom = side * (side * rootdetu + rootdet) # TODO could write using cone.u, check if this is better because rootdetu is derived from u
    @inbounds for i in 1:size(arr, 2)
        @views arr_w = arr[2:end, i]
        @views prod_w = prod[2:end, i]
        Hermitian(svec_to_smat!(cone.work_mat, arr_w, cone.rt2), :U)
        copytri!(cone.work_mat, 'U', cone.is_complex)
        mul!(cone.work_mat2, cone.work_mat, W)
        mul!(cone.work_mat, W, cone.work_mat2, rootdetu * abs2(side), false)

        smat_to_svec!(prod_w, cone.work_mat, cone.rt2)
        prod_w .+= dot(w, arr_w) * rootdet .* w
        prod_w ./= denom
    end
    H11 = abs2(rootdetu) + rootdet * const1
    @. @views prod[1, :] = H11 * arr[1, :]
    @views mul!(prod[1, :]', w', arr[2:end, :], const1, true)
    @views mul!(prod[2:cone.dim, :], w, arr[1, :]', const1, true)
    prod ./= cone.sc_const

    return prod
end

function correction(cone::HypoRootdetTri{T}, primal_dir::AbstractVector{T}) where T
    @assert cone.grad_updated
    u_dir = primal_dir[1]
    @views w_dir = primal_dir[2:end]
    corr = cone.correction
    @views w_corr = corr[2:end]
    sigma = cone.sigma
    z = cone.rootdetu
    tmpw = cone.tmpw

    vec_Wi = smat_to_svec!(tmpw, cone.Wi, cone.rt2)
    S = copytri!(svec_to_smat!(cone.work_mat, w_dir, cone.rt2), 'U', cone.is_complex)
    dot_Wi_S = dot(vec_Wi, w_dir)
    ldiv!(cone.fact_W, S)
    dot_skron = real(dot(S, S'))
    dot_wdwi = dot(vec_Wi, w_dir)

    rdiv!(S, cone.fact_W.U)
    mul!(cone.work_mat2, S, S')
    @views smat_to_svec!(w_corr, cone.work_mat2, cone.rt2)
    w_corr .*= -2 * (sigma + 1)

    ssigma = inv(T(cone.side)) - sigma
    scal1 = sigma * ssigma
    scal2 = dot_Wi_S * (ssigma - sigma)
    udz = u_dir / z
    scal4 = 2 * udz * sigma
    scal5 = scal1 * (dot_skron - scal2 * dot_Wi_S) - scal4 * (scal2 + udz)
    scal6 = 2 * dot_Wi_S * scal1 + scal4
    @. w_corr += scal5 * vec_Wi

    skron2 = rdiv!(S, cone.fact_W.U')
    vec_skron2 = smat_to_svec!(tmpw, skron2, cone.rt2)

    @. w_corr += scal6 * vec_skron2
    corr[1] = (sigma * (dot(vec_skron2, w_dir) - (scal2 + 4 * udz) * dot_wdwi) + 2 * abs2(udz)) / z
    corr .*= cone.sc_const / -2

    return corr
end
