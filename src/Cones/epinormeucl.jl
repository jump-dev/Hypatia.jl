#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(u^2 - norm_2(w)^2)
=#

mutable struct EpiNormEucl{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::AbstractVector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    dist::T

    function EpiNormEucl{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiNormEucl{T}(dim::Int) where {T <: HypReal} = EpiNormEucl{T}(dim, false)

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO maybe only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: HypReal}
    reset_data(cone)
    dim = cone.dim
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
        @. prod[:, j] = ga * cone.grad + aj / cone.dist
        prod[1, j] -= arr[1, j] / disth
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    dist2 = cone.dist * 2
    @inbounds for j in 1:size(prod, 2)
        @views aj = arr[:, j]
        pa = dot(cone.point, aj)
        @. prod[:, j] = pa * cone.point + cone.dist * aj
        prod[1, j] -= arr[1, j] * dist2
    end
    return prod
end

function hess_L_fact(cone::EpiNormEucl{T}) where {T} # remove T
    dist = (abs2(cone.point[1]) - sum(abs2, cone.point[2:end])) # replace by cone.dist when figured out how to double things correctly
    # dist = cone.dist
    dim = cone.dim
    L = zeros(T, dim, dim)
    w = cone.point[2:dim]
    dist_sqrt = sqrt(dist)
    mul!(L[2:dim, 2:dim], w, w')
    # hyp_aat!(L_main, w) # maybe does dispatch to the right thing
    @. L[2:dim, 2:dim] /= (cone.point[1] + dist_sqrt)
    # L_main += dist_sqrt * I # returns an array
    L[2:dim, 2:dim] += dist_sqrt * I
    L[2:dim, 2:dim] = w * w' / (cone.point[1] + dist_sqrt) + dist_sqrt * I
    L[1, 2:dim] = L[2:dim, 1] = -cone.point[2:dim]
    L[1, 1] = cone.point[1]
    @. L /= dist
    @. L *= sqrt(2)
    return L
end

hess_U_fact(cone::EpiNormEucl) = hess_L_fact(cone)

hess_U_prod!(prod, arr, cone::EpiNormEucl) = mul!(prod, hess_L_fact(cone), arr)

function hess_L_div!(dividend, arr, cone::EpiNormEucl{T}) where {T} # remove T
    dist = (abs2(cone.point[1]) - sum(abs2, cone.point[2:end])) # replace by cone.dist when figured out how to double things correctly
    # dist = cone.dist
    dim = cone.dim
    L = zeros(T, dim, dim)
    w = cone.point[2:dim]
    dist_sqrt = sqrt(dist)
    mul!(L[2:dim, 2:dim], w, w')
    # hyp_aat!(L_main, w) # maybe does dispatch to the right thing
    @. L[2:dim, 2:dim] /= (cone.point[1] + dist_sqrt)
    # L_main += dist_sqrt * I # returns an array
    L[2:dim, 2:dim] += dist_sqrt * I
    L[2:dim, 2:dim] = w * w' / (cone.point[1] + dist_sqrt) + dist_sqrt * I
    L[1, 2:dim] = L[2:dim, 1] = cone.point[2:dim]
    L[1, 1] = cone.point[1]
    @. L /= sqrt(2)
    return mul!(dividend, L, arr)
end
