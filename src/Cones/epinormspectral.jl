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

    # -logdet(u * I - W * W' / u) - log(u)
    n = cone.n
    m = cone.m
    W = reshape(cone.point[2:end], cone.n, cone.m)
    u = cone.point[1]
    ui = 1 / u
    X = Symmetric(W * W')
    Z = Symmetric(u * I - X * ui)
    Zi = inv(Z)
    Eu = I + X * ui^2
    cone.g[1] = -sum(Zi .* Eu) - ui
    cone.g[2:end] = vec(2 * Zi * W * ui)

    H11 = sum((Zi * Eu * Zi) .* Eu) + sum(-Zi .* X * (-2 * inv(u^3))) + inv(u^2)
    H1W = -2 * Zi * Eu * Zi * W * inv(u) - 2 * Zi * W * inv(u^2) # correct
    HWW = zeros(cone.m * cone.n, cone.m * cone.n)
    function dzdw(k, l)
        ret = zeros(cone.n, cone.n)
        ret[k, :] -= -inv(u) * W[:, l]
        ret[:, k] -= -inv(u) * W[:, l]
        return ret
    end
    function d2zdw2(ki, li, mi, ni)
        ret = zeros(cone.n, cone.n)
        if li == ni
            ret[ki, mi] -= inv(u)
            ret[mi, ki] -= inv(u)
        end
        return ret
    end
    idx1 = 0
    for j in 1:m, i in 1:n
        idx1 += 1
        idx2 = 0
        for l in 1:m, k in 1:n
            idx2 += 1
            HWW[idx1, idx2] = sum((Zi * dzdw(i, j) * Zi)' .* dzdw(k, l)) - sum(Zi .* d2zdw2(i, j, k, l))
        end
    end

    # HWW = 2 * inv(u) * (Zi .* Zi .* )

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    # cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    # cone.g .= DiffResults.gradient(cone.diffres) # [g1; gW...]
    # cone.H .= DiffResults.hessian(cone.diffres)
    # cone.g = [g1; gW...]
    cone.H = [H11 vec(H1W)'; vec(H1W) HWW]


    # @show H11
    # @show cone.H[1, 1] / H11
    # if !isapprox(cone.H, H)
    #     # @show HWW
    #     # @show cone.H[2:end, 2:end]
    #     @show cone.H[2:end, 2:end] ./ HWW
    # end
    @assert isapprox(cone.H * cone.point, -cone.g, atol = 1e-7, rtol = 1e-7)

    return factorize_hess(cone)
end
