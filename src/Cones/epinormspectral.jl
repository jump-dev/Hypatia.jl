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
    use_dual::Bool
    dim::Int
    n::Int
    m::Int
    is_complex::Bool
    point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    W::Matrix{R}
    Z::Matrix{R}
    fact_Z
    Zi::Hermitian{R, Matrix{R}}
    ZiW::Matrix{R}
    HuW::Matrix{R}
    Huu::T
    tmpmm::Matrix{R}
    tmpnm::Matrix{R}
    tmpnn::Matrix{R}

    function EpiNormSpectral{T, R}(
        n::Int,
        m::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert 1 <= n <= m
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.is_complex = (R <: Complex)
        cone.dim = (cone.is_complex ? 2 * n * m + 1 : n * m + 1)
        cone.n = n
        cone.m = m
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

EpiNormSpectral{T, R}(n::Int, m::Int) where {R <: RealOrComplex{T}} where {T <: Real} = EpiNormSpectral{T, R}(n, m, false)

reset_data(cone::EpiNormSpectral) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.hess_fact_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormSpectral{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.W = Matrix{R}(undef, cone.n, cone.m)
    cone.Z = Matrix{R}(undef, cone.n, cone.n)
    cone.ZiW = Matrix{R}(undef, cone.n, cone.m)
    cone.HuW = Matrix{R}(undef, cone.n, cone.m)
    cone.tmpmm = Matrix{R}(undef, cone.m, cone.m)
    cone.tmpnm = Matrix{R}(undef, cone.n, cone.m)
    cone.tmpnn = Matrix{R}(undef, cone.n, cone.n)
    return
end

get_nu(cone::EpiNormSpectral) = cone.n + 1

function set_initial_point(arr::AbstractVector, cone::EpiNormSpectral{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormSpectral)
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > 0
        @views vec_copy_to!(cone.W[:], cone.point[2:end])
        copyto!(cone.Z, abs2(u) * I) # TODO inefficient
        mul!(cone.Z, cone.W, cone.W', -1, true)
        cone.fact_Z = cholesky!(Hermitian(cone.Z, :U), check = false)
        cone.is_feas = isposdef(cone.fact_Z)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormSpectral)
    @assert cone.is_feas
    u = cone.point[1]

    ldiv!(cone.ZiW, cone.fact_Z, cone.W)
    cone.Zi = Hermitian(inv(cone.fact_Z), :U) # TODO only need trace of inverse here, which we can get from the cholesky factor - if cheap, don't do the inverse until needed in the hessian

    cone.grad[1] = -u * tr(cone.Zi)
    @views vec_copy_to!(cone.grad[2:end], cone.ZiW[:])
    cone.grad .*= 2
    cone.grad[1] += (cone.n - 1) / u

    cone.grad_updated = true
    return cone.grad
end

function update_hess_prod(cone::EpiNormSpectral)
    @assert cone.grad_updated
    u = cone.point[1]
    HuW = cone.HuW

    copyto!(HuW, cone.ZiW)
    HuW .*= -4 * u
    ldiv!(cone.fact_Z, HuW)
    cone.Huu = 4 * abs2(u) * sum(abs2, cone.Zi) + (cone.grad[1] - 2 * (cone.n - 1) / u) / u

    cone.hess_prod_updated = true
    return
end

function update_hess(cone::EpiNormSpectral)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    n = cone.n
    m = cone.m
    u = cone.point[1]
    W = cone.W
    Zi = cone.Zi
    ZiW = cone.ZiW
    tmpmm = cone.tmpmm
    H = cone.hess.data

    # H_W_W part
    mul!(tmpmm, W', ZiW) # TODO Hermitian? W' * Zi * W
    tmpmm += I # TODO inefficient

    # TODO parallelize loops
    idx_incr = (cone.is_complex ? 2 : 1)
    r_idx = 2
    for i in 1:m, j in 1:n
        c_idx = r_idx
        @inbounds for k in i:m
            ZiWjk = ZiW[j, k]
            tmpmmik = tmpmm[i, k]
            lstart = (i == k ? j : 1)
            @inbounds for l in lstart:n
                term1 = Zi[l, j] * tmpmmik
                term2 = ZiW[l, i] * ZiWjk
                _hess_WW_element(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
        r_idx += idx_incr
    end
    H .*= 2

    # H_u_W and H_u_u parts
    @views vec_copy_to!(H[1, 2:end], cone.HuW[:])
    H[1, 1] = cone.Huu

    cone.hess_updated = true
    return cone.hess
end

function _hess_WW_element(H::Matrix{T}, r_idx::Int, c_idx::Int, term1::T, term2::T) where {T <: Real}
    @inbounds H[r_idx, c_idx] = term1 + term2
    return
end

function _hess_WW_element(H::Matrix{T}, r_idx::Int, c_idx::Int, term1::Complex{T}, term2::Complex{T}) where {T <: Real}
    @inbounds begin
        H[r_idx, c_idx] = real(term1) + real(term2)
        H[r_idx + 1, c_idx] = imag(term2) - imag(term1)
        H[r_idx, c_idx + 1] = imag(term1) + imag(term2)
        H[r_idx + 1, c_idx + 1] = real(term1) - real(term2)
    end
    return
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormSpectral)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    u = cone.point[1]
    W = cone.W
    tmpnm = cone.tmpnm
    tmpnn = cone.tmpnn

    @inbounds for j in 1:size(prod, 2)
        arr_1j = arr[1, j]
        @views vec_copy_to!(tmpnm[:], arr[2:end, j])

        prod[1, j] = cone.Huu * arr_1j + real(dot(cone.HuW, tmpnm))

        # prod_2j = 2 * cone.fact_Z \ (((tmpnm * W' + W * tmpnm' - (2 * u * arr_1j) * I) / cone.fact_Z) * W + tmpnm)
        mul!(tmpnn, tmpnm, W')
        @inbounds for j in 1:cone.n
            @inbounds for i in 1:j
                tmpnn[i, j] += tmpnn[j, i]'
            end
            tmpnn[j, j] -= 2 * u * arr_1j
        end
        mul!(tmpnm, Hermitian(tmpnn, :U), cone.ZiW, 2, 2)
        ldiv!(cone.fact_Z, tmpnm)
        @views vec_copy_to!(prod[2:end, j], tmpnm[:])
    end

    return prod
end
