#=
Copyright 2018, Chris Coey and contributors

functions and types for linear objective conic problems of the form:

primal (over x,s):
```
  min  c'x :          duals
    b - Ax == 0       (y)
    h - Gx == s in K  (z)
```
dual (over z,y):
```
  max  -b'y - h'z :      duals
    c + A'y + G'z == 0   (x)
                z in K*  (s)
```
where K is a convex cone defined as a Cartesian product of recognized primitive cones, and K* is its dual cone.

The primal-dual optimality conditions are:
```
         b - Ax == 0
         h - Gx == s
  c + A'y + G'z == 0
            s'z == 0
              s in K
              z in K*
```
=#

mutable struct LinearObjConic <: Model
    n::Int
    p::Int
    q::Int
    c::Vector{Float64}
    A::AbstractMatrix{Float64}
    b::Vector{Float64}
    G::AbstractMatrix{Float64}
    h::Vector{Float64}
    cones::Vector{Cones.Cone}
    cone_idxs::Vector{UnitRange{Int}}
    nu::Float64

    # initial_x::Vector{Float64}
    # initial_y::Vector{Float64}
    # initial_z::Vector{Float64}
    # initial_s::Vector{Float64}

    function LinearObjConic(c, A, b, G, h, cones, cone_idxs)
        model = new()
        model.n = length(c)
        model.p = length(b)
        model.q = length(h)
        model.c = c
        model.A = A
        model.b = b
        model.G = G
        model.h = h
        model.cones = cones
        model.cone_idxs = cone_idxs
        model.nu = isempty(cones) ? 0.0 : sum(Cones.get_nu, cones)
        return model
    end
end

# TODO check model data consistency function

# TODO preprocess function - maybe wraps a model
