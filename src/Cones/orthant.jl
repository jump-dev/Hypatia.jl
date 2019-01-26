#=
Copyright 2018, Chris Coey and contributors

nonnegative/nonpositive orthant cones
nonnegative cone: w in R^n : w_i >= 0
nonpositive cone: w in R^n : w_i <= 0

barriers from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
nonnegative cone: -sum_i(log(u_i))
nonpositive cone: -sum_i(log(-u_i))
=#

mutable struct Nonnegative <: Cone
    usedual::Bool
    dim::Int
    primals::AbstractVector{Float64}

    function Nonnegative(dim::Int, isdual::Bool)
        cone = new()
        cone.usedual = isdual
        cone.dim = dim
        return cone
    end
end

Nonnegative(dim::Int) = Nonnegative(dim, false)
Nonnegative() = Nonnegative(1)

mutable struct Nonpositive <: Cone
    usedual::Bool
    dim::Int
    primals::AbstractVector{Float64}

    function Nonpositive(dim::Int, isdual::Bool)
        cone = new()
        cone.usedual = isdual
        cone.dim = dim
        return cone
    end
end

Nonpositive(dim::Int) = Nonpositive(dim, false)
Nonpositive() = Nonpositive(1)

OrthantCone = Union{Nonnegative, Nonpositive}

get_nu(cone::OrthantCone) = cone.dim

set_initial_point(arr::AbstractVector{Float64}, cone::Nonnegative) = (@. arr = 1.0; arr)
set_initial_point(arr::AbstractVector{Float64}, cone::Nonpositive) = (@. arr = -1.0; arr)

check_in_cone(cone::Nonnegative) = all(u -> (u > 0.0), cone.primals)
check_in_cone(cone::Nonpositive) = all(u -> (u < 0.0), cone.primals)

# function calcg!(g::AbstractVector{Float64}, cone::OrthantCone)
#     @. cone.invpnt = inv(cone.pnt)
#     @. g = -cone.invpnt
#     return g
# end
#
# calcHiarr!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::OrthantCone) = (@. prod = abs2(cone.pnt) * arr; prod)
# calcHarr!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::OrthantCone) = (@. prod = abs2(cone.invpnt) * arr; prod)

grad(cone::OrthantCone) = -inv.(cone.primals)
hess(cone::OrthantCone) = Diagonal(abs2.(inv.(cone.primals)))
inv_hess(cone::OrthantCone) = Diagonal(abs2.(cone.primals))

# function get_max_alpha(cone::Nonnegative, direction::AbstractVector{Float64})
#     if all(u -> (u > 0.0), direction)
#         return Inf
#     end
#     return -maximum(cone.primals[l] / direction[l] for l in eachindex(direction) if direction[l] < 0.0)
# end
#
# function get_max_alpha(cone::Nonpositive, direction::AbstractVector{Float64})
#     if all(u -> (u < 0.0), direction)
#         return Inf
#     end
#     return minimum(cone.primals[l] / direction[l] for l in eachindex(direction) if direction[l] > 0.0)
# end
