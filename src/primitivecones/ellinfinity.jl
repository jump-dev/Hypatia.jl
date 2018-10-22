#=
Copyright 2018, Chris Coey and contributors

epigraph of L-infinity norm
(u, v) : u >= ||v||_inf for u in R, v in R^n
barrier is -sum_j ln (u^2 - v_j^2) + (n-1) ln u
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
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::EllInfinityCone) = (@. arr = 0.0; arr[1] = 1.0; arr)
loadpnt_prmtv!(prmtv::EllInfinityCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::EllInfinityCone)
    u = prmtv.pnt[1]
    v = view(prmtv.pnt, 2:prmtv.dim)
    if u <= maximum(abs, v)
        return false
    end

    # TODO don't explicitly construct full matrix
    g = prmtv.g
    H = prmtv.H
    usqr = abs2(u)
    g1 = 0.0
    h1 = 0.0
    for j in eachindex(v)
        vj = 2.0/(usqr - abs2(v[j]))
        g1 += vj
        wj = vj*v[j]
        h1 += abs2(vj)
        jp1 = j + 1
        g[jp1] = wj
        H[jp1,jp1] = vj + abs2(wj)
        H[1,jp1] = H[jp1,1] = -vj*wj*u
    end
    invu = inv(u)
    t1 = (prmtv.dim - 2)*invu
    g[1] = t1 - u*g1
    H[1,1] = -t1*invu + usqr*h1 - g1

    @. prmtv.H2 = prmtv.H
    prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
    return issuccess(prmtv.F)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::EllInfinityCone) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EllInfinityCone) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EllInfinityCone) = mul!(prod, prmtv.H, arr)
