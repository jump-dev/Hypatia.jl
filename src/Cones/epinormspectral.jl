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

    H11 = sum((Zi * Eu * Zi) .* Eu) + sum(-Zi .* X * (-2 * ui^3)) + ui^2
    H1W = -2 * Zi * Eu * Zi * W * ui - 2 * Zi * W * ui^2
    HWW = zeros(cone.m * cone.n, cone.m * cone.n)

    idx1 = 0
    for j in 1:m, i in 1:n
        idx1 += 1
        dzdwij = zeros(n, n)
        dzdwij[i, :] += W[:, j] * ui
        dzdwij[:, i] += W[:, j] * ui
        term1 = (Zi * dzdwij * Zi)

        idx2 = idx1 - 1
        # l = j
        for k in i:n
            idx2 += 1
            dzdwkl = zeros(n, n)
            dzdwkl[k, :] += W[:, j] * ui
            dzdwkl[:, k] += W[:, j] * ui
            HWW[idx1, idx2] = sum(term1 .* dzdwkl) + 2 * ui * Zi[i, k]
        end


        for l in (j + 1):m, k in 1:n
            idx2 += 1
            dzdwkl = zeros(n, n)
            dzdwkl[k, :] += W[:, l] * ui
            dzdwkl[:, k] += W[:, l] * ui
            HWW[idx1, idx2] = sum(term1 .* dzdwkl)
        end
    end


    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    # cone.g .= DiffResults.gradient(cone.diffres) # [g1; gW...]
    cone.H .= DiffResults.hessian(cone.diffres)
    # cone.g = [g1; gW...]
    H = [H11 vec(H1W)'; vec(H1W) Symmetric(HWW, :U)]


    # @show H11
    # @show cone.H[1, 1] / H11
    if !isapprox(cone.H, H)
        # @show HWW
        # @show cone.H[2:end, 2:end]
        @show cone.H[2:end, 2:end] ./ HWW
    end
    @assert isapprox(cone.H * cone.point, -cone.g, atol = 1e-6, rtol = 1e-6)

    return factorize_hess(cone)
end
