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

    is_feas::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    grad::Vector{T}
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}

    function Nonnegative{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

Nonnegative{T}(dim::Int) where {T <: HypReal} = Nonnegative{T}(dim, false)
# Nonnegative{T}() where {T <: HypReal} = Nonnegative{T}(1)

mutable struct Nonpositive{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::AbstractVector{T}

    is_feas::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    grad::Vector{T}
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}

    function Nonpositive{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

Nonpositive{T}(dim::Int) where {T <: HypReal} = Nonpositive{T}(dim, false)
# Nonpositive{T}() where {T <: HypReal} = Nonpositive{T}(1)

const OrthantCone{T <: HypReal} = Union{Nonnegative{T}, Nonpositive{T}}

# TODO maybe only allocate the fields we use
function setup_data(cone::OrthantCone{T}) where {T <: HypReal}
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Diagonal(zeros(T, dim), :U)
    cone.inv_hess = Diagonal(zeros(T, dim), :U)
    return
end

get_nu(cone::OrthantCone) = cone.dim

set_initial_point(arr::AbstractVector, cone::Nonnegative) = (arr .= 1)
set_initial_point(arr::AbstractVector, cone::Nonpositive) = (arr .= -1)

reset_data(cone::OrthantCone) = (cone.is_feas = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function update_feas(cone::Nonnegative)
    @assert !cone.is_feas
    cone.is_feas = all(u -> (u > 0), cone.point)
    return cone.is_feas
end
function update_feas(cone::Nonpositive)
    @assert !cone.is_feas
    cone.is_feas = all(u -> (u < 0), cone.point)
    return cone.is_feas
end

function update_grad(cone::OrthantCone)
    @assert cone.is_feas
    @. cone.grad = -inv(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::OrthantCone)
    @assert cone.grad_updated
    @. cone.hess.diag = abs2(cone.grad)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::OrthantCone)
    @assert cone.is_feas
    @. cone.inv_hess.diag = abs2(cone.point)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod() = nothing
update_inv_hess_prod() = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::OrthantCone)
    @assert cone.is_feas
    @. prod = arr / cone.point / cone.point
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::OrthantCone)
    @assert cone.is_feas
    @. prod = arr * cone.point * cone.point
    return prod
end
