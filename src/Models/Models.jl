#=
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
where K is a convex cone defined as a Cartesian product of recognized proper
cones, and K* is its dual cone.
An objective offset can be provided as the keyword arg `obj_offset` (default 0).

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

module Models

import Hypatia.Cones

mutable struct Model{T <: Real}
    n::Int
    p::Int
    q::Int
    obj_offset::T
    c::Vector{T}
    A
    b::Vector{T}
    G
    h::Vector{T}
    cones::Vector{Cones.Cone{T}}
    cone_idxs::Vector{UnitRange{Int}}
    nu::T

    function Model{T}(
        c::Vector{T},
        A,
        b::Vector{T},
        G,
        h::Vector{T},
        cones::Vector{Cones.Cone{T}};
        obj_offset::T = zero(T),
        ) where {T <: Real}
        model = new{T}()

        model.n = length(c)
        model.p = length(b)
        model.q = length(h)
        model.obj_offset = obj_offset
        model.c = c
        model.A = A
        model.b = b
        model.G = G
        model.h = h
        model.cones = cones
        model.cone_idxs = build_cone_idxs(model.q, model.cones)
        model.nu = isempty(cones) ? zero(T) : sum(Cones.get_nu, cones)

        return model
    end
end

function build_cone_idxs(q::Int, cones::Vector{Cones.Cone{T}}) where {T <: Real}
    cone_idxs = Vector{UnitRange{Int}}(undef, length(cones))
    prev_idx = 0
    for (k, cone) in enumerate(cones)
        dim = Cones.dimension(cone)
        cone_idxs[k] = (prev_idx + 1):(prev_idx + dim)
        prev_idx += dim
    end
    @assert q == prev_idx
    return cone_idxs
end

get_cone_idxs(model::Model) = model.cone_idxs

# make the model's A and G matrices dense
function densify!(model::Model{T}) where {T <: Real}
    model.A = convert(Matrix{T}, model.A)
    model.G = convert(Matrix{T}, model.G)
    return model
end

end
