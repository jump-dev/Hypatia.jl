#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of matrix spectral norm (operator norm associated with standard Euclidean norm; i.e. maximum singular value)
(u in R, W in R^{n,m}) : u >= opnorm(W)
note n <= m is enforced WLOG since opnorm(W) = opnorm(W')
W is vectorized column-by-column (i.e. vec(W) in Julia)

barrier from "Interior-Point Polynomial Algorithms in Convex Programming" by Nesterov & Nemirovskii 1994
-logdet(u*I_n - W*W'/u) - log(u)
= -logdet(u^2*I_n - W*W') + (n - 1) log(u)
=#

mutable struct EpiNormSpectral{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    d1::Int
    d2::Int
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    scal_hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    scal_hess
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    W::Matrix{R}
    Z::Matrix{R}
    fact_Z
    Zi::Hermitian{R, Matrix{R}}
    tau::Matrix{R}
    HuW::Matrix{R}
    Huu::T
    trZi2::T
    Wtau::Matrix{R}
    Zitau::Matrix{R}
    tmpd1d2::Matrix{R}
    tmpd1d1::Matrix{R}
    tmpd2d2::Matrix{R}

    function EpiNormSpectral{T, R}(
        d1::Int,
        d2::Int;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert 1 <= d1 <= d2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.is_complex = (R <: Complex)
        cone.dim = (cone.is_complex ? 2 * d1 * d2 + 1 : d1 * d2 + 1)
        cone.d1 = d1
        cone.d2 = d2
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

reset_data(cone::EpiNormSpectral) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.scal_hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.hess_fact_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormSpectral{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.scal_hess = zeros(T, dim, dim)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.W = Matrix{R}(undef, cone.d1, cone.d2)
    cone.Z = Matrix{R}(undef, cone.d1, cone.d1)
    cone.tau = Matrix{R}(undef, cone.d1, cone.d2)
    cone.HuW = Matrix{R}(undef, cone.d1, cone.d2)
    cone.Wtau = Matrix{R}(undef, cone.d2, cone.d2)
    cone.Zitau = Matrix{R}(undef, cone.d1, cone.d2)
    cone.tmpd1d2 = Matrix{R}(undef, cone.d1, cone.d2)
    cone.tmpd1d1 = Matrix{R}(undef, cone.d1, cone.d1)
    cone.tmpd2d2 = Matrix{R}(undef, cone.d2, cone.d2)
    return
end

get_nu(cone::EpiNormSpectral) = cone.d1 + 1

use_correction(cone::EpiNormSpectral) = true

function set_initial_point(arr::AbstractVector, cone::EpiNormSpectral{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormSpectral{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views vec_copy_to!(cone.W, cone.point[2:end])
        copyto!(cone.Z, abs2(u) * I)
        mul!(cone.Z, cone.W, cone.W', -1, true)
        cone.fact_Z = cholesky!(Hermitian(cone.Z, :U), check = false)
        cone.is_feas = isposdef(cone.fact_Z)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

# TODO is there a faster way to check u >= nuc_norm, eg thru a cholesky?
# update_dual_feas(cone::EpiNormSpectral) = true
function update_dual_feas(cone::EpiNormSpectral{T}) where {T}
    u = cone.dual_point[1]
    if u > eps(T)
        W = @views vec_copy_to!(similar(cone.W), cone.dual_point[2:end])
        return (u - sum(svdvals(W)) > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormSpectral)
    @assert cone.is_feas
    u = cone.point[1]

    ldiv!(cone.tau, cone.fact_Z, cone.W)
    cone.Zi = Hermitian(inv(cone.fact_Z), :U) # TODO only need trace of inverse here, which we can get from the cholesky factor - if cheap, don't do the inverse until needed in the hessian
    cone.grad[1] = -u * tr(cone.Zi)
    @views vec_copy_to!(cone.grad[2:end], cone.tau)
    cone.grad .*= 2
    cone.grad[1] += (cone.d1 - 1) / u

    cone.grad_updated = true
    return cone.grad
end

function update_hess_prod(cone::EpiNormSpectral)
    @assert cone.grad_updated
    u = cone.point[1]
    Zitau = cone.Zitau

    copyto!(Zitau, cone.tau)
    ldiv!(cone.fact_Z, Zitau)
    @. cone.HuW = -4 * u * Zitau
    cone.trZi2 = sum(abs2, cone.Zi)
    cone.Huu = 4 * abs2(u) * cone.trZi2 + (cone.grad[1] - 2 * (cone.d1 - 1) / u) / u

    cone.hess_prod_updated = true
    return
end

function update_hess(cone::EpiNormSpectral)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    d1 = cone.d1
    d2 = cone.d2
    u = cone.point[1]
    W = cone.W
    Zi = cone.Zi
    tau = cone.tau
    Wtau = cone.Wtau
    H = cone.hess.data

    # H_W_W part
    copyto!(Wtau, I)
    mul!(Wtau, W', tau, true, true)

    # TODO parallelize loops
    idx_incr = (cone.is_complex ? 2 : 1)
    r_idx = 2
    for i in 1:d2, j in 1:d1
        c_idx = r_idx
        @inbounds for k in i:d2
            taujk = tau[j, k]
            Wtauik = Wtau[i, k]
            lstart = (i == k ? j : 1)
            @inbounds for l in lstart:d1
                term1 = Zi[l, j] * Wtauik
                term2 = tau[l, i] * taujk
                hess_element(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
        r_idx += idx_incr
    end
    H .*= 2

    # H_u_W and H_u_u parts
    @views vec_copy_to!(H[1, 2:end], cone.HuW)
    H[1, 1] = cone.Huu

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormSpectral)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    u = cone.point[1]
    W = cone.W
    tmpd1d2 = cone.tmpd1d2
    tmpd1d1 = cone.tmpd1d1

    @inbounds for j in 1:size(prod, 2)
        arr_1j = arr[1, j]
        @views vec_copy_to!(tmpd1d2, arr[2:end, j])

        prod[1, j] = cone.Huu * arr_1j + real(dot(cone.HuW, tmpd1d2))

        mul!(tmpd1d1, tmpd1d2, W')
        @inbounds for k in 1:cone.d1
            @inbounds for i in 1:k
                tmpd1d1[i, k] += tmpd1d1[k, i]'
            end
            tmpd1d1[k, k] -= 2 * u * arr_1j
        end
        mul!(tmpd1d2, Hermitian(tmpd1d1, :U), cone.tau, 2, 2)
        ldiv!(cone.fact_Z, tmpd1d2)
        @views vec_copy_to!(prod[2:end, j], tmpd1d2)
    end

    return prod
end

function correction2(cone::EpiNormSpectral, primal_dir::AbstractVector)
    @assert cone.hess_updated

    u = cone.point[1]
    W = cone.W
    Zi = cone.Zi
    tau = cone.tau
    Zitau = cone.Zitau
    WtauI = cone.Wtau
    u_dir = primal_dir[1]
    tmpd1d2 = cone.tmpd1d2
    tmpd1d1 = cone.tmpd1d1
    tmpd2d2 = cone.tmpd2d2
    @views W_dir = vec_copy_to!(tmpd1d2, primal_dir[2:end])
    corr = cone.correction

    Wdtau = mul!(tmpd2d2, W_dir', tau)
    ZiWd = cone.fact_Z \ W_dir
    ZiWdWtauI = ZiWd * WtauI

    Wtmp1 = -2 * u * (cone.fact_Z \ ZiWdWtauI + (ZiWd * W' + tau * W_dir') * Zitau + Zitau * Wdtau)
    Wtmp2 = 4 * u * u_dir * u * Zitau - u_dir * tau
    ldiv!(cone.fact_Z, Wtmp2)
    Wtmp1 += Wtmp2

    Wcorr = tau * (Wdtau * Wdtau + W_dir' * ZiWdWtauI) + ZiWdWtauI * Wdtau + ZiWd * Wdtau' * WtauI + u_dir * Wtmp1
    Wcorr .*= -2
    @views vec_copy_to!(corr[2:end], Wcorr)

    ZiLi = cone.fact_Z.L \ Zi
    trZi3 = sum(abs2, ZiLi)
    corr[1] = -real(dot(W_dir, Wtmp1 + 3 * Wtmp2)) - u * u_dir * (6 * cone.trZi2 - 8 * u * trZi3 * u) * u_dir - (cone.d1 - 1) * abs2(u_dir / u) / u

    return corr
end
