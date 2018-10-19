#=
Copyright 2018, Chris Coey and contributors

epigraph of spectral norm, or operator norm of matrix X in R^{n x m} associated with standard euclidean norms on R^m and R^n i.e. (y, X) : y >= opnorm(X)
note n <= m is enforced WLOG since opnorm(X) = opnorm(X')
X is vectorized column-by-column (i.e. vec(X) in Julia)
barrier for matrix cone is -ln det(y*I_n - X*X'/y) - ln y
from Nesterov & Nemirovskii 1994 "Interior-Point Polynomial Algorithms in Convex Programming"

# TODO don't need ForwardDiff: use identity for inverse of matrix plus I to get inv(y*I - X*X'/y) using properties of SVD unitary matrices U and V'
=#

mutable struct SpectralNormCone <: PrimitiveCone
    dim::Int
    n::Int
    m::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64} # TODO could be faster as StaticArray
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function SpectralNormCone(dim::Int, n::Int, m::Int)
        @assert dim == n*m + 1
        @assert n <= m
        prmtv = new()
        prmtv.dim = dim
        prmtv.n = n
        prmtv.m = m
        prmtv.mat = Matrix{Float64}(undef, n, m)
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        function barfun(x)
            X = reshape(x[2:end], n, m)
            y = x[1]
            mat = y*I - X*X'/y
            return -logdet(mat) - log(y)
        end
        prmtv.barfun = barfun
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

dimension(prmtv::SpectralNormCone) = prmtv.dim
barrierpar_prmtv(prmtv::SpectralNormCone) = prmtv.n + 1
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::SpectralNormCone) = (@. arr = 0.0; arr[1] = 1.0; arr)
loadpnt_prmtv!(prmtv::SpectralNormCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::SpectralNormCone)
    prmtv.mat[:] = @view prmtv.pnt[2:end] # TODO a little slow
    F = svd!(prmtv.mat) # TODO reduce allocs further
    if F.S[1] >= prmtv.pnt[1]
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.pnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)

    @. prmtv.H2 = prmtv.H
    prmtv.F = cholesky!(Symmetric(prmtv.H2), Val(true), check=false) # bunchkaufman if it fails
    if !isposdef(prmtv.F)
        @. prmtv.H2 = prmtv.H
        prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
        return issuccess(prmtv.F)
    end
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::SpectralNormCone) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::SpectralNormCone) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::SpectralNormCone) = mul!(prod, prmtv.H, arr)
