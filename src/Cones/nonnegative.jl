"""
$(TYPEDEF)

Nonnegative cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""
mutable struct Nonnegative{T <: Real} <: Cone{T}
    dim::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}

    function Nonnegative{T}(dim::Int) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        return cone
    end
end

use_dual_barrier(::Nonnegative) = false

reset_data(cone::Nonnegative) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = false)

use_sqrt_hess_oracles(::Int, cone::Nonnegative) = true

get_nu(cone::Nonnegative) = cone.dim

set_initial_point!(arr::AbstractVector, cone::Nonnegative) = (arr .= 1)

function update_feas(cone::Nonnegative{T}) where T
    @assert !cone.feas_updated
    cone.is_feas = all(>(eps(T)), cone.point)
    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::Nonnegative{T}) where T = all(>(eps(T)), cone.dual_point)

function update_grad(cone::Nonnegative)
    @assert cone.is_feas
    @. cone.grad = -inv(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Nonnegative{T}) where T
    cone.grad_updated || update_grad(cone)
    if !isdefined(cone, :hess)
        cone.hess = Diagonal(zeros(T, cone.dim))
    end

    @. cone.hess.diag = abs2(cone.grad)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::Nonnegative{T}) where T
    @assert cone.is_feas
    if !isdefined(cone, :inv_hess)
        cone.inv_hess = Diagonal(zeros(T, cone.dim))
    end

    @. cone.inv_hess.diag = abs2(cone.point)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Nonnegative,
    )
    @assert cone.is_feas
    @. prod = arr / cone.point / cone.point
    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Nonnegative,
    )
    @assert cone.is_feas
    @. prod = arr * cone.point * cone.point
    return prod
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Nonnegative,
    )
    @assert cone.is_feas
    @. prod = arr / cone.point
    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Nonnegative,
    )
    @assert cone.is_feas
    @. prod = arr * cone.point
    return prod
end

function dder3(cone::Nonnegative, dir::AbstractVector)
    @. cone.dder3 = abs2(dir / cone.point) / cone.point
    return cone.dder3
end

hess_nz_count(cone::Nonnegative) = cone.dim
hess_nz_count_tril(cone::Nonnegative) = cone.dim
inv_hess_nz_count(cone::Nonnegative) = cone.dim
inv_hess_nz_count_tril(cone::Nonnegative) = cone.dim
hess_nz_idxs_col(cone::Nonnegative, j::Int) = [j]
hess_nz_idxs_col_tril(cone::Nonnegative, j::Int) = [j]
inv_hess_nz_idxs_col(cone::Nonnegative, j::Int) = [j]
inv_hess_nz_idxs_col_tril(cone::Nonnegative, j::Int) = [j]

# nonnegative is not primitive, so sum and max proximity measures differ
function get_proximity(
    cone::Nonnegative{T},
    rtmu::T,
    use_sum_prox::Bool, # use sum proximity
    negtol::T = sqrt(eps(T)),
    ) where {T <: Real}
    proxs = (abs(si * zi / rtmu - 1) for (si, zi) in
        zip(cone.point, cone.dual_point))
    if use_sum_prox
        return sum(proxs)
    else
        return maximum(proxs)
    end
end
