#=
Copyright 2018, Chris Coey and contributors

epigraph of matrix spectral norm (operator norm associated with standard Euclidean norm; i.e. maximum singular value)
(u in R, W in R^{n,m}) : u >= opnorm(W)
note n <= m is enforced WLOG since opnorm(W) = opnorm(W')
W is vectorized column-by-column (i.e. vec(W) in Julia)

barrier from "Interior-Point Polynomial Algorithms in Convex Programming" by Nesterov & Nemirovskii 1994
-logdet(u*I_n - W*W'/u) - log(u)

TODO eliminate allocations
=#

mutable struct EpiNormSpectral{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    n::Int
    m::Int

    point::AbstractVector{T}
    W::Matrix{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    F

    function EpiNormSpectral{T}(n::Int, m::Int, is_dual::Bool) where {T <: HypReal}
        @assert n <= m
        dim = n * m + 1
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.n = n
        cone.m = m
        return cone
    end
end

EpiNormSpectral{T}(n::Int, m::Int) where {T <: HypReal} = EpiNormSpectral{T}(n, m, false)

function setup_data(cone::EpiNormSpectral{T}) where {T <: HypReal}
    dim = cone.dim
    cone.W = Matrix{T}(undef, cone.n, cone.m)
    cone.g = Vector{T}(undef, dim)
    cone.H = zeros(T, dim, dim)
    cone.H2 = similar(cone.H)
    return
end

get_nu(cone::EpiNormSpectral) = cone.n + 1

set_initial_point(arr::AbstractVector{T}, cone::EpiNormSpectral{T}) where {T <: HypReal} = (@. arr = zero(T); arr[1] = one(T); arr)

function check_in_cone(cone::EpiNormSpectral{T}) where {T <: HypReal}
    TimerOutputs.@timeit to "epibar" begin
    u = cone.point[1]
    if u <= zero(T)
        return false
    end
    W = cone.W
    W[:] = view(cone.point, 2:cone.dim)
    n = cone.n
    m = cone.m

    X = Symmetric(W * W') # TODO use syrk
    Z = Symmetric(u * I - X / u)
    F = hyp_chol!(Z)
    if !isposdef(F)
        return false
    end

    # TODO figure out structured form of inverse? could simplify algebra
    Zi = Symmetric(inv(F))
    Eu = Symmetric(I + X / u / u)
    cone.H .= zero(T)

    cone.g[1] = -dot(Zi, Eu) - inv(u)
    cone.g[2:end] = vec(2 * Zi * W / u)

    ZiEuZi = Symmetric(Zi * Eu * Zi)
    cone.H[1, 1] = dot(ZiEuZi, Eu) + (2 * dot(Zi, X) / u + one(T)) / u / u
    cone.H[1, 2:end] = vec(-2 * (ZiEuZi + Zi / u) *  W / u)

    tmpvec = zeros(n)
    tmpmat = zeros(n, n)

    p = 2
    TimerOutputs.@timeit to "barloop" for j in 1:m, i in 1:n
        for d in 1:n
            tmpvec[d] = sum(W[c, j] / u * Zi[c, d] for c in 1:n)
            for c in 1:n
                tmpmat[c, d] = Zi[c, i] * tmpvec[d]
            end
        end

        # Zi * dZdWij * Zi
        TimerOutputs.@timeit to "syrk" term1 = Symmetric(tmpmat + tmpmat') # TODO use syrk

        TimerOutputs.@timeit to "matrixify" begin
        q = p
        # l = j
        # @inbounds for k in i:n
        #     TimerOutputs.@timeit to "sum1" cone.H[p, q] += 2 * (sum(term1[ni, k] * W[ni, j] for ni in 1:n) + Zi[i, k]) / u
        #     q += 1
        # end

        cone.H[p, q:(q + n - i)] = Zi[i, i:n]
        for ni in 1:n
            cone.H[p, q:(q + n - i)] += term1[ni, i:n] * W[ni, j]
        end
        cone.H[p, q:(q + n - i)] *= 2 / u
        q += (n - i + 1)

        # gives exactly same # iters as loop code
        # arr = 2 * (sum(term1[ni, i:n]' * W[ni, j] for ni in 1:n) .+ Zi[i, i:n]') / u
        # cone.H[p, q:(q + n - i)] += arr[:]
        # q += (n - i + 1)

        # @inbounds for l in (j + 1):m, k in 1:n
        #     TimerOutputs.@timeit to "sum2" cone.H[p, q] += 2 * sum(term1[ni, k] * W[ni, l] for ni in 1:n) / u
        #     q += 1
        # end

        # gives exactly same # iters as loop code
        mat = 2 * term1 * W[:, (j + 1):m] / u
        nterms = n * (m - j)
        cone.H[p, q:(q + nterms - 1)] += mat[:]
        q += nterms

        p += 1

        end #mat
    end
    end # timeit

    return factorize_hess(cone)
end
