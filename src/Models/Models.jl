#=
Copyright 2018, Chris Coey and contributors

functions and types for model data
=#

module Models

import Hypatia.Cones

abstract type Model end

mutable struct Point
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    s::Vector{Float64}

    z_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
    s_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
    dual_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
    primal_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}

    function Point(model::Model)
        point = new()

        point.x = zeros(length(model.c))
        point.y = zeros(length(model.b))
        point.z = zeros(length(model.h))
        point.s = zeros(length(model.h))

        point.z_views = [view(point.z, idxs) for idxs in model.cone_idxs]
        point.s_views = [view(point.s, idxs) for idxs in model.cone_idxs]
        point.dual_views = [Cones.use_dual(model.cones[k]) ? point.s_views[k] : point.z_views[k] for k in eachindex(model.cones)]
        point.primal_views = [Cones.use_dual(model.cones[k]) ? point.z_views[k] : point.s_views[k] for k in eachindex(model.cones)]

        return point
    end
end

include("linear.jl")

# include("smooth_convex.jl") # TODO convex quadratic or smooth nonlinear objectives

end
