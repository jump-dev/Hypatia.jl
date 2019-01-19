#=
Copyright 2018, Chris Coey and contributors

epigraph of perspective of (half) square function (AKA rotated second-order cone)
(u in R, v in R_+, w in R^n) : u >= v*1/2*norm_2(w/v)^2
note v*1/2*norm_2(w/v)^2 = 1/2*sum_i(w_i^2)/v

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(2*u*v - norm_2(w)^2)
=#

mutable struct EpiPerSquare <: PrimitiveCone
    usedual::Bool
    dim::Int
    pnt::AbstractVector{Float64}
    dist::Float64
    Hi::Matrix{Float64}
    H::Matrix{Float64}

    function EpiPerSquare(dim::Int, isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        prmtv.Hi = Matrix{Float64}(undef, dim, dim)
        prmtv.H = similar(prmtv.Hi)
        return prmtv
    end
end

EpiPerSquare(dim::Int) = EpiPerSquare(dim, false)

dimension(prmtv::EpiPerSquare) = prmtv.dim
barrierpar_prmtv(prmtv::EpiPerSquare) = 2
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::EpiPerSquare) = (@. arr = 0.0; arr[1] = 1.0; arr[2] = 1.0; arr)
loadpnt_prmtv!(prmtv::EpiPerSquare, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::EpiPerSquare, scal::Float64)
    u = prmtv.pnt[1]
    v = prmtv.pnt[2]
    w = view(prmtv.pnt, 3:prmtv.dim)
    if u <= 0.0 || v <= 0.0
        return false
    end
    nrm2 = 0.5*sum(abs2, w)
    dist = u*v - nrm2
    if dist <= 0.0
        return false
    end
    prmtv.dist = dist

    H = prmtv.H
    Hi = prmtv.Hi
    mul!(Hi, prmtv.pnt, prmtv.pnt')
    Hi[2,1] = Hi[1,2] = nrm2
    for j in 3:prmtv.dim
        Hi[j,j] += dist
    end
    @. H = Hi
    for j in 3:prmtv.dim
        H[1,j] = H[j,1] = -Hi[2,j]
        H[2,j] = H[j,2] = -Hi[1,j]
    end
    H[1,1] = Hi[2,2]
    H[2,2] = Hi[1,1]
    @. H *= abs2(inv(dist))
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::EpiPerSquare) = (@. g = prmtv.pnt/prmtv.dist; tmp = g[1]; g[1] = -g[2]; g[2] = -tmp; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EpiPerSquare) = mul!(prod, prmtv.Hi, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EpiPerSquare) = mul!(prod, prmtv.H, arr)
