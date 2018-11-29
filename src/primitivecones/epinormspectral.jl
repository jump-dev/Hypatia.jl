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

mutable struct EpiNormSpectral <: PrimitiveCone
    usedual::Bool
    dim::Int
    n::Int
    m::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function EpiNormSpectral(n::Int, m::Int, isdual::Bool)
        @assert n <= m
        dim = n*m + 1
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        prmtv.n = n
        prmtv.m = m
        prmtv.mat = Matrix{Float64}(undef, n, m)
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        function barfun(pnt)
            W = reshape(pnt[2:end], n, m)
            u = pnt[1]
            return -logdet(u*I - W*W'/u) - log(u)
        end
        prmtv.barfun = barfun
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

EpiNormSpectral(n::Int, m::Int) = EpiNormSpectral(n, m, false)

dimension(prmtv::EpiNormSpectral) = prmtv.dim
barrierpar_prmtv(prmtv::EpiNormSpectral) = prmtv.n + 1
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::EpiNormSpectral) = (@. arr = 0.0; arr[1] = 1.0; arr)
loadpnt_prmtv!(prmtv::EpiNormSpectral, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::EpiNormSpectral, scal::Float64)
    prmtv.mat[:] = @view prmtv.pnt[2:end] # TODO a little slow
    F = svd!(prmtv.mat) # TODO reduce allocs further
    if F.S[1] >= prmtv.pnt[1]
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.pnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)

    return factH(prmtv)
end
