#=
Copyright 2018, Chris Coey and contributors

nonnegative/nonpositive orthant cones
from Nesterov & Todd "Self-Scaled Barriers and Interior-Point Methods for Convex Programming"
=#

# nonnegative cone barrier is -sum_j ln u_j
mutable struct NonnegativeCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    invpnt::Vector{Float64}

    function NonnegativeCone(dim::Int)
        prmtv = new()
        prmtv.dim = dim
        prmtv.invpnt = Vector{Float64}(undef, dim)
        return prmtv
    end
end

# nonpositive cone barrier is -sum_j ln u_j
mutable struct NonpositiveCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    invpnt::Vector{Float64}

    function NonpositiveCone(dim::Int)
        prmtv = new()
        prmtv.dim = dim
        prmtv.invpnt = Vector{Float64}(undef, dim)
        return prmtv
    end
end

OrthantCone = Union{NonnegativeCone, NonpositiveCone}

dimension(prmtv::OrthantCone) = prmtv.dim
barrierpar_prmtv(prmtv::OrthantCone) = prmtv.dim

getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::NonnegativeCone) = (@. arr = 1.0; arr)
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::NonpositiveCone) = (@. arr = -1.0; arr)

loadpnt_prmtv!(prmtv::OrthantCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

incone_prmtv(prmtv::NonnegativeCone) = all(u -> (u > 0.0), prmtv.pnt)
incone_prmtv(prmtv::NonpositiveCone) = all(u -> (u < 0.0), prmtv.pnt)

function calcg_prmtv!(g::AbstractVector{Float64}, prmtv::OrthantCone)
    @. prmtv.invpnt = inv(prmtv.pnt)
    @. g = -prmtv.invpnt
    return g
end

calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::OrthantCone) = (@. prod = abs2(prmtv.pnt)*arr; prod)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::OrthantCone) = (@. prod = abs2(prmtv.invpnt)*arr; prod)
