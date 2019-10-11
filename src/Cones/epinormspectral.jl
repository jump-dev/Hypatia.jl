#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of matrix spectral norm (operator norm associated with standard Euclidean norm; i.e. maximum singular value)
(u in R, W in R^{n,m}) : u >= opnorm(W)
note n <= m is enforced WLOG since opnorm(W) = opnorm(W')
W is vectorized column-by-column (i.e. vec(W) in Julia)

barrier from "Interior-Point Polynomial Algorithms in Convex Programming" by Nesterov & Nemirovskii 1994
-logdet(u*I_n - W*W'/u) - log(u)
=#

mutable struct EpiNormSpectral{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    n::Int
    m::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    W::Matrix{T}
    Z::Matrix{T}
    fact_Z
    Zi::Symmetric{T, Matrix{T}}
    tmpmm::Matrix{T}
    tmpnm::Matrix{T}

    function EpiNormSpectral{T}(
        n::Int,
        m::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert 1 <= n <= m
        dim = n * m + 1
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.n = n
        cone.m = m
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

EpiNormSpectral{T}(n::Int, m::Int) where {T <: Real} = EpiNormSpectral{T}(n, m, false)

function setup_data(cone::EpiNormSpectral{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.W = Matrix{T}(undef, cone.n, cone.m)
    cone.Z = Matrix{T}(undef, cone.n, cone.n)
    cone.tmpmm = Matrix{T}(undef, cone.m, cone.m)
    cone.tmpnm = Matrix{T}(undef, cone.n, cone.m)
    return
end

get_nu(cone::EpiNormSpectral) = cone.n + 1

function set_initial_point(arr::AbstractVector, cone::EpiNormSpectral)
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::EpiNormSpectral)
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > 0
        cone.W[:] .= view(cone.point, 2:cone.dim)
        copyto!(cone.Z, abs2(u) * I)
        mul!(cone.Z, cone.W, cone.W', -1, true)
        cone.fact_Z = cholesky!(Symmetric(cone.Z, :U), check = false)
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

    ldiv!(cone.tmpnm, cone.fact_Z, cone.W)
    cone.Zi = Symmetric(inv(cone.fact_Z), :U) # TODO only need trace of inverse here, which we can get from the cholesky factor - if cheap, don't do the inverse until needed in the hessian

    cone.grad[1] = -u * tr(cone.Zi)
    cone.grad[2:end] = cone.tmpnm
    cone.grad .*= 2
    cone.grad[1] += (cone.n - 1) / u

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiNormSpectral)
    @assert cone.grad_updated
    n = cone.n
    m = cone.m
    u = cone.point[1]
    Zi = cone.Zi
    tmpmm = cone.tmpmm
    tmpnm = cone.tmpnm
    H = cone.hess.data

    # H_W_W part
    mul!(tmpmm, cone.W', tmpnm) # symmetric, W' * Zi * W. TODO calculate using ldiv with L then syrk?
    # TODO parallelize loops
    for i in 1:m
        r = 1 + (i - 1) * n
        for j in 1:n
            r2 = r + j
            @. @views @inbounds H[r2, r .+ (j:n)] = Zi[j:n, j] * tmpmm[i, i] + tmpnm[j:n, i] * tmpnm[j, i] + Zi[j, j:n]
            c2 = r + n
            for k in (i + 1):m
                @. @views @inbounds H[r2, c2 .+ (1:n)] = Zi[1:n, j] * tmpmm[i, k] + tmpnm[1:n, i] * tmpnm[j, k]
                c2 += n
            end
        end
    end
    H .*= 2

    # H_u_W part (careful: modifies tmpnm)
    ldiv!(cone.fact_Z, tmpnm)
    tmpnm .*= -4u
    H[1, 2:end] = tmpnm

    # H_u_u part
    H[1, 1] = 4 * abs2(u) * sum(abs2, Zi) + (cone.grad[1] - 2 * (n - 1) / u) / u

    cone.hess_updated = true
    return cone.hess
end

# TODO try to get inverse hessian using analogy to epinormeucl barrier
