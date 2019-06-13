#=
Copyright 2018, Chris Coey and contributors

functions and types for model data
=#

module Models

using LinearAlgebra
using SparseArrays
import Hypatia.Cones
import Hypatia.HypReal

mutable struct Point{T <: HypReal}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    s::Vector{T}

    z_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    s_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    dual_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    primal_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}

    function Point(
        x::Vector{T},
        y::Vector{T},
        z::Vector{T},
        s::Vector{T},
        cones::Vector{<:Cones.Cone{T}},
        cone_idxs::Vector{UnitRange{Int}},
        ) where {T <: HypReal}
        point = new{T}()

        point.x = x
        point.y = y
        point.z = z
        point.s = s

        point.z_views = [view(point.z, idxs) for idxs in cone_idxs]
        point.s_views = [view(point.s, idxs) for idxs in cone_idxs]
        point.dual_views = [Cones.use_dual(cones[k]) ? point.s_views[k] : point.z_views[k] for k in eachindex(cones)]
        point.primal_views = [Cones.use_dual(cones[k]) ? point.z_views[k] : point.s_views[k] for k in eachindex(cones)]

        return point
    end
end

abstract type Model{T <: HypReal} end

abstract type LinearModel{T <: HypReal} <: Model{T} end
include("linear.jl")

# TODO other model types eg quadratic obj, convex differentiable obj

end
