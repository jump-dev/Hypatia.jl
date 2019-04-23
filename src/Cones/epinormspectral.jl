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
    cone.mat[:] = view(cone.point, 2:cone.dim) # TODO a little slow
    F = svd!(cone.mat) # TODO reduce allocs further
    if F.S[1] >= cone.point[1]
        return false
    end

    # -logdet(u * I - W * W' / u) - log(u)
    n = cone.n
    m = cone.m
    W = reshape(cone.point[2:end], cone.n, cone.m)
    u = cone.point[1]
    X = Symmetric(W * W')
    Z = Symmetric(u * I - X / u)
    Zi = Symmetric(inv(Z))
    Eu = Symmetric(I + X / u^2)

    cone.g[1] = -dot(Zi, Eu) - 1 / u
    cone.g[2:end] = vec(2 * Zi * W / u)

    ZiEuZi = Symmetric(Zi * Eu * Zi)
    cone.H[1, 1] = dot(ZiEuZi, Eu) + (2 * dot(Zi, X) + u) / u / u / u
    cone.H[1, 2:end] = vec((ZiEuZi * u + Zi) * -2 *  W / u / u)

    idx1 = 0
    for j in 1:m, i in 1:n
        idx1 += 1
        dzdwij = zeros(n, n)
        dzdwij[i, :] += W[:, j] / u
        dzdwij[:, i] += W[:, j] / u
        term1 = Zi * dzdwij * Zi

        idx2 = idx1 - 1
        # l = j
        for k in i:n
            idx2 += 1
            dzdwkl = zeros(n, n)
            dzdwkl[k, :] += W[:, j] / u
            dzdwkl[:, k] += W[:, j] / u
            cone.H[idx1 + 1, idx2 + 1] = dot(term1, dzdwkl) + 2 * Zi[i, k] / u
        end

        for l in (j + 1):m, k in 1:n
            idx2 += 1
            dzdwkl = zeros(n, n)
            dzdwkl[k, :] += W[:, l] / u
            dzdwkl[:, k] += W[:, l] / u
            cone.H[idx1 + 1, idx2 + 1] = dot(term1, dzdwkl)
        end
    end

    @assert isapprox(Symmetric(cone.H, :U) * cone.point, -cone.g, atol = 1e-7, rtol = 1e-7) # TODO remove

    return factorize_hess(cone)
end
