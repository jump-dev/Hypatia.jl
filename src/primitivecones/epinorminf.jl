#=
Copyright 2018, Chris Coey and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u^2 - w_i^2)) + (n-1)*log(u)

TODO for efficiency, don't construct full H matrix (arrow fill)
=#

mutable struct EpiNormInf <: PrimitiveCone
    usedual::Bool
    dim::Int
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F

    function EpiNormInf(dim::Int, isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        @. prmtv.H = 0.0
        prmtv.H2 = copy(prmtv.H)
        return prmtv
    end
end

EpiNormInf(dim::Int) = EpiNormInf(dim, false)

dimension(prmtv::EpiNormInf) = prmtv.dim
barrierpar_prmtv(prmtv::EpiNormInf) = prmtv.dim
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::EpiNormInf) = (@. arr = 0.0; arr[1] = 1.0; arr)
loadpnt_prmtv!(prmtv::EpiNormInf, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::EpiNormInf)
    u = prmtv.pnt[1]
    w = view(prmtv.pnt, 2:prmtv.dim)
    if u <= maximum(abs, w)
        return false
    end

    # TODO don't explicitly construct full matrix
    g = prmtv.g
    H = prmtv.H
    usqr = abs2(u)
    g1 = 0.0
    h1 = 0.0
    for j in eachindex(w)
        iuwj = 2.0/(usqr - abs2(w[j]))
        g1 += iuwj
        wiuwj = w[j]*iuwj
        h1 += abs2(iuwj)
        jp1 = j + 1
        g[jp1] = wiuwj
        H[jp1,jp1] = iuwj + abs2(wiuwj)
        H[1,jp1] = H[jp1,1] = -iuwj*wiuwj*u
    end
    invu = inv(u)
    t1 = (prmtv.dim - 2)*invu
    g[1] = t1 - u*g1
    H[1,1] = -t1*invu + usqr*h1 - g1

    return factH(prmtv)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::EpiNormInf) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EpiNormInf) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EpiNormInf) = mul!(prod, prmtv.H, arr)
