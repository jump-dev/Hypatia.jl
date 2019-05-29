#=
Copyright 2018, Chris Coey and contributors

nonnegative/nonpositive orthant cones
nonnegative cone: w in R^n : w_i >= 0
nonpositive cone: w in R^n : w_i <= 0

barriers from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
nonnegative cone: -sum_i(log(u_i))
nonpositive cone: -sum_i(log(-u_i))
=#

mutable struct Nonnegative{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int

    point::AbstractVector{T}

    function Nonnegative{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

Nonnegative{T}(dim::Int) where {T <: HypReal} = Nonnegative{T}(dim, false)
Nonnegative{T}() where {T <: HypReal} = Nonnegative{T}(1)

mutable struct Nonpositive{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int

    point::AbstractVector{T}

    function Nonpositive{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

Nonpositive{T}(dim::Int) where {T <: HypReal} = Nonpositive{T}(dim, false)
Nonpositive{T}() where {T <: HypReal} = Nonpositive{T}(1)

const OrthantCone{T <: HypReal} = Union{Nonnegative{T}, Nonpositive{T}}

setup_data(cone::OrthantCone) = nothing

get_nu(cone::OrthantCone) = cone.dim

set_initial_point(arr::AbstractVector{T}, cone::Nonnegative{T}) where {T <: HypReal} = (@. arr = one(T); arr)
set_initial_point(arr::AbstractVector{T}, cone::Nonpositive{T}) where {T <: HypReal} = (@. arr = -one(T); arr)

check_in_cone(cone::Nonnegative{T}) where {T <: HypReal} = all(u -> (u > zero(T)), cone.point)
check_in_cone(cone::Nonpositive{T}) where {T <: HypReal} = all(u -> (u < zero(T)), cone.point)

# TODO eliminate allocs
grad(cone::OrthantCone) = -inv.(cone.point)
hess(cone::OrthantCone) = Diagonal(abs2.(inv.(cone.point)))
inv_hess(cone::OrthantCone) = Diagonal(abs2.(cone.point))

hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::OrthantCone{T}) where {T <: HypReal} = (@. prod = arr / cone.point / cone.point; prod)
inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::OrthantCone{T}) where {T <: HypReal} = (@. prod = arr * cone.point * cone.point; prod)
