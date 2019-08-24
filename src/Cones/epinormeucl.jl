#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(u^2 - norm_2(w)^2)
=#

mutable struct EpiNormEucl{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    dist::T

    function EpiNormEucl{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiNormEucl{T}(dim::Int) where {T <: Real} = EpiNormEucl{T}(dim, false)

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO maybe only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

get_nu(cone::EpiNormEucl) = 2

function set_initial_point(arr::AbstractVector, cone::EpiNormEucl)
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::EpiNormEucl)
    @assert !cone.feas_updated
    u = cone.point[1]
    if u > 0
        w = view(cone.point, 2:cone.dim)
        cone.dist = (abs2(u) - sum(abs2, w)) / 2
        cone.is_feas = (cone.dist > 0)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormEucl)
    @assert cone.is_feas
    @. cone.grad = cone.point / cone.dist
    cone.grad[1] *= -1
    cone.grad_updated = true
    return cone.grad
end

# TODO only work with upper triangle
function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    mul!(cone.hess.data, cone.grad, cone.grad')
    cone.hess += inv(cone.dist) * I
    cone.hess[1, 1] -= 2 / cone.dist
    cone.hess_updated = true
    return cone.hess
end

# TODO only work with upper triangle
function update_inv_hess(cone::EpiNormEucl)
    @assert cone.is_feas
    mul!(cone.inv_hess.data, cone.point, cone.point')
    cone.inv_hess += cone.dist * I
    cone.inv_hess[1, 1] -= 2 * cone.dist
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod(cone::EpiNormEucl) = nothing
update_inv_hess_prod(cone::EpiNormEucl) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.grad_updated
    disth = cone.dist / 2
    @inbounds for j in 1:size(prod, 2)
        @views aj = arr[:, j]
        ga = dot(cone.grad, aj)
        @. prod[:, j] = ga * cone.grad
    end
    @. prod += arr / cone.dist
    # mul!(prod, cone.grad)
    prod[1, :] -= arr[1, :] / disth
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    @views @inbounds for j in 1:size(prod, 2)
        aj = arr[:, j]
        pa = dot(cone.point, aj)
        @. prod[:, j] = pa * cone.point
    end
    @. prod += cone.dist * arr
    @views @. prod[1, :] -= arr[1, :] * cone.dist * 2
    return prod
end
