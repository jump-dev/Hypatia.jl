#=
Copyright 2018, Chris Coey and contributors

nonnegative/nonpositive orthant cones
from Nesterov & Todd "Self-Scaled Barriers and Interior-Point Methods for Convex Programming"
=#

# nonnegative cone barrier is -sum_j ln x_j
mutable struct NonnegativeCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    invpnt::Vector{Float64}

    function NonnegativeCone(dim::Int)
        prm = new()
        prm.dim = dim
        prm.invpnt = Vector{Float64}(undef, dim)
        return prm
    end
end

# nonpositive cone barrier is -sum_j ln x_j
mutable struct NonpositiveCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    invpnt::Vector{Float64}

    function NonpositiveCone(dim::Int)
        prm = new()
        prm.dim = dim
        prm.invpnt = Vector{Float64}(undef, dim)
        return prm
    end
end

OrthantCone = Union{NonnegativeCone, NonpositiveCone}

dimension(prm::OrthantCone) = prm.dim
barrierpar_prm(prm::OrthantCone) = prm.dim

getintdir_prm!(arr::AbstractVector{Float64}, prm::NonnegativeCone) = (@. arr = 1.0; arr)
getintdir_prm!(arr::AbstractVector{Float64}, prm::NonpositiveCone) = (@. arr = -1.0; arr)

loadpnt_prm!(prm::OrthantCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

incone_prm(prm::NonnegativeCone) = all(x -> (x > 0.0), prm.pnt)
incone_prm(prm::NonpositiveCone) = all(x -> (x < 0.0), prm.pnt)

function calcg_prm!(g::AbstractVector{Float64}, prm::OrthantCone)
    @. prm.invpnt = inv(prm.pnt)
    @. g = -prm.invpnt
    return g
end

calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::OrthantCone) = (@. prod = abs2(prm.pnt)*arr; prod)
calcHarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::OrthantCone) = (@. prod = abs2(prm.invpnt)*arr; prod)
