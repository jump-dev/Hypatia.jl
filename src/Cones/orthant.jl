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
    use_dual::Bool
    dim::Int

    point::AbstractVector{Float64}

    function Nonnegative(dim::Int, is_dual::Bool)
        cone = new()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

Nonnegative(dim::Int) = Nonnegative(dim, false)
Nonnegative() = Nonnegative(1)

mutable struct Nonpositive <: Cone
    use_dual::Bool
    dim::Int

    point::AbstractVector{Float64}

    function Nonpositive(dim::Int, is_dual::Bool)
        cone = new()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

Nonpositive(dim::Int) = Nonpositive(dim, false)
Nonpositive() = Nonpositive(1)

OrthantCone = Union{Nonnegative, Nonpositive}

setup_data(cone::OrthantCone) = nothing

get_nu(cone::OrthantCone) = cone.dim

set_initial_point(arr::AbstractVector{Float64}, cone::Nonnegative) = (@. arr = 1.0; arr)
set_initial_point(arr::AbstractVector{Float64}, cone::Nonpositive) = (@. arr = -1.0; arr)

check_in_cone(cone::Nonnegative) = all(u -> (u > 0.0), cone.point)
check_in_cone(cone::Nonpositive) = all(u -> (u < 0.0), cone.point)

grad(cone::OrthantCone) = -inv.(cone.point)
hess(cone::OrthantCone) = Diagonal(abs2.(inv.(cone.point)))
inv_hess(cone::OrthantCone) = Diagonal(abs2.(cone.point))

hess_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::OrthantCone) = (@. prod = arr / cone.point / cone.point; prod)
inv_hess_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::OrthantCone) = (@. prod = arr * cone.point * cone.point; prod)
