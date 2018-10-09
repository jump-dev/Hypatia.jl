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
        prm = new()
        prm.dim = dim
        prm.g = Vector{Float64}(undef, dim)
        prm.H = similar(prm.g, dim, dim)
        @. prm.H = 0.0
        prm.H2 = copy(prm.H)
        return prm
    end
end

dimension(prm::EllInfinityCone) = prm.dim
barrierpar_prm(prm::EllInfinityCone) = prm.dim
getintdir_prm!(arr::AbstractVector{Float64}, prm::EllInfinityCone) = (arr[1] = 1.0; @. arr[2:end] = 0.0; arr)
loadpnt_prm!(prm::EllInfinityCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::EllInfinityCone)
    if prm.pnt[1] <= maximum(abs, prm.pnt[j] for j in 2:prm.dim)
        return false
    end

    g = prm.g
    H = prm.H
    x = prm.pnt[1]
    xsqr = abs2(x)
    g1 = 0.0
    h1 = 0.0
    for j in 2:prm.dim
        vj = 2.0/(xsqr - abs2(prm.pnt[j]))
        g1 += vj
        wj = vj*prm.pnt[j]
        h1 += abs2(vj)
        g[j] = wj
        H[j,j] = vj + abs2(wj)
        H[1,j] = H[j,1] = -vj*wj*x
    end
    invx = inv(x)
    t1 = (prm.dim - 2)*invx
    g[1] = t1 - x*g1
    H[1,1] = -t1*invx + xsqr*h1 - g1

    @. prm.H2 = H
    prm.F = cholesky!(Symmetric(prm.H2), check=false) # bunchkaufman if it fails
    if !issuccess(prm.F)
        @. prm.H2 = H
        prm.F = bunchkaufman!(Symmetric(prm.H2), check=false)
    end
    return issuccess(prm.F)
end

calcg_prm!(g::AbstractVector{Float64}, prm::EllInfinityCone) = (@. g = prm.g; g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::EllInfinityCone) = ldiv!(prod, prm.F, arr)
calcHarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::EllInfinityCone) = mul!(prod, prm.H, arr)
