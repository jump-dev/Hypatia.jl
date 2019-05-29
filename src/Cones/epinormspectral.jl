#=
Copyright 2018, Chris Coey and contributors

epigraph of matrix spectral norm (operator norm associated with standard Euclidean norm; i.e. maximum singular value)
(u in R, W in R^{n,m}) : u >= opnorm(W)
note n <= m is enforced WLOG since opnorm(W) = opnorm(W')
W is vectorized column-by-column (i.e. vec(W) in Julia)

barrier from "Interior-Point Polynomial Algorithms in Convex Programming" by Nesterov & Nemirovskii 1994
-logdet(u*I_n - W*W'/u) - log(u)

# TODO don't use ForwardDiff: use identity for inverse of matrix plus I and properties of SVD unitary matrices
# TODO eliminate allocations for incone check
=#

mutable struct EpiNormSpectral{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    n::Int
    m::Int
    
    point::AbstractVector{Float64}
    W::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F

    function EpiNormSpectral(n::Int, m::Int, is_dual::Bool)
        @assert n <= m
        dim = n * m + 1
        cone = new()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.n = n
        cone.m = m
        return cone
    end
end

EpiNormSpectral(n::Int, m::Int) = EpiNormSpectral(n, m, false)

function setup_data(cone::EpiNormSpectral)
    dim = cone.dim
    cone.W = Matrix{Float64}(undef, cone.n, cone.m)
    cone.g = Vector{Float64}(undef, dim)
    cone.H = Matrix{Float64}(undef, dim, dim)
    cone.H2 = similar(cone.H)
    return
end

get_nu(cone::EpiNormSpectral) = cone.n + 1

set_initial_point(arr::AbstractVector{Float64}, cone::EpiNormSpectral) = (@. arr = 0.0; arr[1] = 1.0; arr)

function check_in_cone(cone::EpiNormSpectral)
    u = cone.point[1]
    if u <= 0.0
        return false
    end
    W = cone.W
    W[:] = view(cone.point, 2:cone.dim)
    n = cone.n
    m = cone.m
    X = Symmetric(W * W')
    Z = Symmetric(u * I - X / u)
    F = cholesky(Z, Val(true), check = false) # TODO in place
    if !isposdef(F)
        return false
    end
    # TODO figure out structured form of inverse? could simplify algebra
    Zi = Symmetric(inv(F))
    Eu = Symmetric(I + X / u^2)
    cone.H .= 0.0

    cone.g[1] = -dot(Zi, Eu) - 1 / u
    cone.g[2:end] = vec(2 * Zi * W / u)

    ZiEuZi = Symmetric(Zi * Eu * Zi)
    cone.H[1, 1] = dot(ZiEuZi, Eu) + (2 * dot(Zi, X) / u + 1) / u / u
    cone.H[1, 2:end] = vec(-2 * (ZiEuZi + Zi / u) *  W / u)

    tmpvec = zeros(n)
    tmpmat = zeros(n, n)

    p = 2
    for j in 1:m, i in 1:n
        for d in 1:n
            tmpvec[d] = sum(W[c, j] / u * Zi[c, d] for c in 1:n)
            for c in 1:n
                tmpmat[c, d] = Zi[c, i] * tmpvec[d]
            end
        end

        # Zi * dZdWij * Zi
        term1 = Symmetric(tmpmat + tmpmat')

        # TODO matrixify
        q = p
        # l = j
        for k in i:n
            cone.H[p, q] += 2 * (sum(term1[ni, k] * W[ni, j] for ni in 1:n) + Zi[i, k]) / u
            q += 1
        end
        for l in (j + 1):m, k in 1:n
            cone.H[p, q] += 2 * sum(term1[ni, k] * W[ni, l] for ni in 1:n) / u
            q += 1
        end
        p += 1
    end

    # @assert isapprox(Symmetric(cone.H, :U) * cone.point, -cone.g, atol = 1e-6, rtol = 1e-6)

    return factorize_hess(cone)
end
