#=
Copyright 2018, Chris Coey and contributors

nonnegative/nonpositive orthant cones
nonnegative cone: w in R^n : w_i >= 0
nonpositive cone: w in R^n : w_i <= 0

barriers from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
nonnegative cone: -sum_i(log(u_i))
nonpositive cone: -sum_i(log(-u_i))
=#

mutable struct Nonnegative{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}

    function Nonnegative{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

Nonnegative{T}(dim::Int) where {T <: Real} = Nonnegative{T}(dim, false)
# Nonnegative{T}() where {T <: Real} = Nonnegative{T}(1)

mutable struct Nonpositive{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}

    function Nonpositive{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

Nonpositive{T}(dim::Int) where {T <: Real} = Nonpositive{T}(dim, false)
# Nonpositive{T}() where {T <: Real} = Nonpositive{T}(1)

const OrthantCone{T <: Real} = Union{Nonnegative{T}, Nonpositive{T}}

reset_data(cone::OrthantCone) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO maybe only allocate the fields we use
function setup_data(cone::OrthantCone{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Diagonal(zeros(T, dim))
    cone.inv_hess = Diagonal(zeros(T, dim))
    return
end

get_nu(cone::OrthantCone) = cone.dim

set_initial_point(arr::AbstractVector, cone::Nonnegative) = (arr .= 1)
set_initial_point(arr::AbstractVector, cone::Nonpositive) = (arr .= -1)

function update_feas(cone::Nonnegative)
    @assert !cone.feas_updated
    cone.is_feas = all(u -> (u > 0), cone.point)
    cone.feas_updated = true
    return cone.is_feas
end
function update_feas(cone::Nonpositive)
    @assert !cone.feas_updated
    cone.is_feas = all(u -> (u < 0), cone.point)
    cone.feas_updated = true
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

update_hess_prod(cone::OrthantCone) = nothing
update_inv_hess_prod(cone::OrthantCone) = nothing

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

hess_nnzs(cone::OrthantCone, ::Bool) = cone.dim

# hess_sparsity_pattern(cone::OrthantCone{T}) where {T <: Real} = sparse(one(T) * I, cone.dim, cone.dim)

hess_nz_idxs_j(cone::OrthantCone, j::Int, ::Bool) = j:j
inv_hess_nz_idxs_j(cone::OrthantCone, j::Int, b::Bool) = hess_nz_idxs_j(cone, j, b)
