#=
Copyright 2018, Chris Coey and contributors

epigraph of operator norm of matrix X in R^{n x m} associated with standard euclidean norms on R^m and R^n i.e. (y, X) : y >= opnorm(X)
note n <= m is enforced WLOG since opnorm(X) = opnorm(X')
X is vectorized column-by-column (i.e. vec(X) in Julia)
barrier for matrix cone is -ln det(y*I_n - X*X'/y) - ln y
from Nesterov & Nemirovskii 1994 "Interior-Point Polynomial Algorithms in Convex Programming"
=#

mutable struct OperatorNormCone <: PrimitiveCone
    dim::Int
    n::Int
    m::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64} # TODO could be faster as StaticArray
    H2::Matrix{Float64}
    F

    function OperatorNormCone(dim::Int, n::Int, m::Int)
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
        return prmtv
    end
end

dimension(prmtv::OperatorNormCone) = prmtv.dim
barrierpar_prmtv(prmtv::OperatorNormCone) = prmtv.n + 1
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::OperatorNormCone) = (@. arr = 0.0; arr[1] = 1.0; arr)
loadpnt_prmtv!(prmtv::OperatorNormCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::OperatorNormCone)
    prmtv.mat[:] = @view prmtv.pnt[2:end] # TODO a little slow
    F = svd!(prmtv.mat) # TODO reduce allocs further
    if F.S[1] >= prmtv.pnt[1]
        return false
    end

    # TODO don't need ForwardDiff: use identity for inverse of matrix plus I to get inv(y*I - X*X'/y) using properties of SVD unitary matrices U and V'


    # TODO verify against ForwardDiff g and H
    # -ln det(y*I_n - X*X'/y) - ln y
    function f(x)
        X = reshape(x[2:end], prmtv.n, prmtv.m)
        y = x[1]
        mat = y*I - X*X'/y
        return -logdet(mat) - log(y)
    end
    # TODO more efficient to cache FD object like in power cone
    fdgrad = ForwardDiff.gradient(f, prmtv.pnt)
    fdhess = ForwardDiff.hessian(f, prmtv.pnt)

    @. prmtv.g = fdgrad
    @. prmtv.H = fdhess
    @. prmtv.H2 = prmtv.H
    prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
    return issuccess(prmtv.F)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::OperatorNormCone) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::OperatorNormCone) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::OperatorNormCone) = mul!(prod, prmtv.H, arr)
