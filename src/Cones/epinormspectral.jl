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

mutable struct EpiNormSpectral <: Cone
    use_dual::Bool
    dim::Int
    n::Int
    m::Int
    point::AbstractVector{Float64}
    mat::Matrix{Float64}
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
        cone.mat = Matrix{Float64}(undef, n, m)
        cone.g = Vector{Float64}(undef, dim)
        cone.H = Matrix{Float64}(undef, dim, dim)
        cone.H2 = similar(cone.H)
        return cone
    end
end

EpiNormSpectral(n::Int, m::Int) = EpiNormSpectral(n, m, false)

get_nu(cone::EpiNormSpectral) = cone.n + 1

set_initial_point(arr::AbstractVector{Float64}, cone::EpiNormSpectral) = (@. arr = 0.0; arr[1] = 1.0; arr)

function check_in_cone(cone::EpiNormSpectral)
    u = cone.point[1]
    if u <= 0
        return false
    end
    cone.mat[:] = view(cone.point, 2:cone.dim) # TODO a little slow
    n = cone.n
    m = cone.m
    W = cone.mat
    X = Symmetric(W * W')
    Z = Symmetric(u * I - X / u)
    F = cholesky(Z, Val(true), check = false) # TODO in place
    if !isposdef(F)
        return false
    end
    Zi = Symmetric(inv(F))
    Eu = Symmetric(I + X / u^2)

    cone.g[1] = -dot(Zi, Eu) - 1 / u
    cone.g[2:end] = vec(2 * Zi * W / u)

    ZiEuZi = Symmetric(Zi * Eu * Zi)
    cone.H[1, 1] = dot(ZiEuZi, Eu) + (2 * dot(Zi, X) / u + 1) / u / u
    cone.H[1, 2:end] = vec((ZiEuZi + Zi / u) * -2 *  W / u)

    p = 2
    for j in 1:m, i in 1:n
        dzdwij = zeros(n, n)
        dzdwij[i, :] += W[:, j] / u
        V = Zi * dzdwij * Zi
        term1 = Symmetric(V + V')

        q = p - 1
        # l = j
        for k in i:n
            q += 1
            cone.H[p, q] = 2 * (sum(term1[ni, k] * W[ni, j] for ni in 1:n) + Zi[i, k]) / u
        end
        for l in (j + 1):m, k in 1:n
            q += 1
            cone.H[p, q] = 2 * sum(term1[ni, k] * W[ni, l] for ni in 1:n) / u
        end
        p += 1
    end

    @assert isapprox(Symmetric(cone.H, :U) * cone.point, -cone.g, atol = 1e-7, rtol = 1e-7) # TODO remove

    return factorize_hess(cone)
end
