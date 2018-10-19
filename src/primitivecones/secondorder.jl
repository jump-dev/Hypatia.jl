#=
Copyright 2018, Chris Coey and contributors

second order cone
(x, y) : x >= norm(y)
barrier is -ln(x^2 - norm(y)^2)
from Nesterov & Todd "Self-Scaled Barriers and Interior-Point Methods for Convex Programming"
=#

mutable struct SecondOrderCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    dist::Float64
    Hi::Matrix{Float64}
    H::Matrix{Float64}

    function SecondOrderCone(dim::Int)
        prmtv = new()
        prmtv.dim = dim
        prmtv.Hi = Matrix{Float64}(undef, dim, dim)
        prmtv.H = similar(prmtv.Hi)
        return prmtv
    end
end

dimension(prmtv::SecondOrderCone) = prmtv.dim
barrierpar_prmtv(prmtv::SecondOrderCone) = 1
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::SecondOrderCone) = (@. arr = 0.0; arr[1] = 1.0; arr)
loadpnt_prmtv!(prmtv::SecondOrderCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::SecondOrderCone)
    if prmtv.pnt[1] <= 0
        return false
    end
    prmtv.dist = abs2(prmtv.pnt[1]) - sum(abs2, prmtv.pnt[j] for j in 2:prmtv.dim)
    if prmtv.dist <= 0.0
        return false
    end

    mul!(prmtv.Hi, prmtv.pnt, prmtv.pnt')
    @. prmtv.Hi += prmtv.Hi
    prmtv.Hi[1,1] -= prmtv.dist
    for j in 2:prmtv.dim
        prmtv.Hi[j,j] += prmtv.dist
    end
    @. prmtv.H = prmtv.Hi
    for j in 2:prmtv.dim
        prmtv.H[1,j] = prmtv.H[j,1] = -prmtv.H[j,1]
    end
    @. prmtv.H *= inv(prmtv.dist)^2
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::SecondOrderCone) = (@. g = prmtv.pnt/prmtv.dist; g[1] = -g[1]; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::SecondOrderCone) = mul!(prod, prmtv.Hi, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::SecondOrderCone) = mul!(prod, prmtv.H, arr)
