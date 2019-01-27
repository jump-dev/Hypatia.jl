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
    barfun::Function
    diffres

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
        function barfun(point)
            W = reshape(point[2:end], n, m)
            u = point[1]
            return -logdet(u * I - W * W' / u) - log(u)
        end
        cone.barfun = barfun
        cone.diffres = DiffResults.HessianResult(cone.g)
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

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)

    return factorize_hess(cone)
end
