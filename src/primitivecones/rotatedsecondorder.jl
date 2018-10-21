#=
Copyright 2018, Chris Coey and contributors

rotated second order cone
(u, v, w) : 2*u*v >= norm(w)^2, u,v >= 0
barrier is -ln(2*u*v - norm(w)^2)
from Nesterov & Todd "Self-Scaled Barriers and Interior-Point Methods for Convex Programming"
=#

mutable struct RotatedSecondOrderCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    disth::Float64
    Hi::Matrix{Float64}
    H::Matrix{Float64}

    function RotatedSecondOrderCone(dim::Int)
        prmtv = new()
        prmtv.dim = dim
        prmtv.Hi = Matrix{Float64}(undef, dim, dim)
        prmtv.H = similar(prmtv.Hi)
        return prmtv
    end
end

dimension(prmtv::RotatedSecondOrderCone) = prmtv.dim
barrierpar_prmtv(prmtv::RotatedSecondOrderCone) = 2
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::RotatedSecondOrderCone) = (@. arr = 0.0; arr[1] = 1.0; arr[2] = 1.0; arr)
loadpnt_prmtv!(prmtv::RotatedSecondOrderCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::RotatedSecondOrderCone)
    pnt = prmtv.pnt
    if (pnt[1] <= 0) || (pnt[2] <= 0)
        return false
    end
    nrm2 = 0.5*sum(abs2, pnt[j] for j in 3:prmtv.dim)
    prmtv.disth = pnt[1]*pnt[2] - nrm2
    if prmtv.disth <= 0.0
        return false
    end

    mul!(prmtv.Hi, pnt, pnt')
    prmtv.Hi[2,1] = prmtv.Hi[1,2] = nrm2
    for j in 3:prmtv.dim
        prmtv.Hi[j,j] += prmtv.disth
    end
    @. prmtv.H = prmtv.Hi
    for j in 3:prmtv.dim
        prmtv.H[1,j] = prmtv.H[j,1] = -prmtv.Hi[2,j]
        prmtv.H[2,j] = prmtv.H[j,2] = -prmtv.Hi[1,j]
    end
    prmtv.H[1,1] = prmtv.Hi[2,2]
    prmtv.H[2,2] = prmtv.Hi[1,1]
    @. prmtv.H *= inv(prmtv.disth)^2
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::RotatedSecondOrderCone) = (@. g = prmtv.pnt/prmtv.disth; tmp = g[1]; g[1] = -g[2]; g[2] = -tmp; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::RotatedSecondOrderCone) = mul!(prod, prmtv.Hi, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::RotatedSecondOrderCone) = mul!(prod, prmtv.H, arr)
