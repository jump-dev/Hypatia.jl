#=
Copyright 2018, Chris Coey and contributors

epigraph of L-infinity norm
(x, y) : x >= ||y||_inf for x in R, y in R^n
barrier is -sum_j ln (x^2 - y_j^2) + (n-1) ln x
from "Barrier Functions in Interior Point Methods" by Osman Guler
=#

mutable struct EllInfinityCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F

    function EllInfinityCone(dim::Int)
        prmtv = new()
        prmtv.dim = dim
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        @. prmtv.H = 0.0
        prmtv.H2 = copy(prmtv.H)
        return prmtv
    end
end

dimension(prmtv::EllInfinityCone) = prmtv.dim
barrierpar_prmtv(prmtv::EllInfinityCone) = prmtv.dim
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::EllInfinityCone) = (arr[1] = 1.0; @. arr[2:end] = 0.0; arr)
loadpnt_prmtv!(prmtv::EllInfinityCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::EllInfinityCone)
    if prmtv.pnt[1] <= maximum(abs, prmtv.pnt[j] for j in 2:prmtv.dim)
        return false
    end

    g = prmtv.g
    H = prmtv.H
    x = prmtv.pnt[1]
    xsqr = abs2(x)
    g1 = 0.0
    h1 = 0.0
    for j in 2:prmtv.dim
        vj = 2.0/(xsqr - abs2(prmtv.pnt[j]))
        g1 += vj
        wj = vj*prmtv.pnt[j]
        h1 += abs2(vj)
        g[j] = wj
        H[j,j] = vj + abs2(wj)
        H[1,j] = H[j,1] = -vj*wj*x
    end
    invx = inv(x)
    t1 = (prmtv.dim - 2)*invx
    g[1] = t1 - x*g1
    H[1,1] = -t1*invx + xsqr*h1 - g1

    @. prmtv.H2 = prmtv.H
    prmtv.F = cholesky!(Symmetric(prmtv.H2), Val(true), check=false) # bunchkaufman if it fails
    if !isposdef(prmtv.F)
        @. prmtv.H2 = prmtv.H
        prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
        return issuccess(prmtv.F)
    end
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::EllInfinityCone) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EllInfinityCone) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EllInfinityCone) = mul!(prod, prmtv.H, arr)
