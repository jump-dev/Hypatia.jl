#=
Copyright 2018, Chris Coey and contributors

epigraph of perspective of (half) square function (AKA rotated second-order cone)
(u in R, v in R_+, w in R^n) : u >= v*1/2*norm_2(w/v)^2
note v*1/2*norm_2(w/v)^2 = 1/2*sum_i(w_i^2)/v

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(2*u*v - norm_2(w)^2)
=#

mutable struct EpiPerSquare{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::AbstractVector{T}

    is_feas::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    dist::T

    function EpiPerSquare{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiPerSquare{T}(dim::Int) where {T <: HypReal} = EpiPerSquare{T}(dim, false)

function setup_data(cone::EpiPerSquare{T}) where {T <: HypReal}
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

get_nu(cone::EpiPerSquare) = 2

function set_initial_point(arr::AbstractVector, cone::EpiPerSquare)
    arr .= 0
    arr[1:2] .= 1
    return arr
end

reset_data(cone::EpiPerSquare) = (cone.is_feas = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function update_feas(cone::EpiPerSquare)
    @assert !cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    if u > 0 && v > 0
        w = view(cone.point, 3:cone.dim)
        cone.dist = u * v - sum(abs2, w) / 2
        cone.is_feas = (cone.dist > 0)
    end
    return cone.is_feas
end

function update_grad(cone::EpiPerSquare)
    @assert cone.is_feas
    @. cone.grad = cone.point / cone.dist
    g2 = cone.grad[2]
    cone.grad[2] = -cone.grad[1]
    cone.grad[1] = -g2
    cone.grad_updated = true
    return cone.grad
end

# TODO only work with upper triangle
function update_hess(cone::EpiPerSquare)
    @assert cone.grad_updated
    mul!(cone.hess.data, cone.point, cone.point')
    invdist = inv(cone.dist)
    for j in 3:cone.dim
        cone.hess.data[j, j] += invdist
        cone.hess.data[1, j] *= -1
        cone.hess.data[2, j] *= -1
    end
    cone.hess.data[1, 2] -= invdist
    cone.hess_updated = true
    return cone.hess
end

# TODO only work with upper triangle
function update_inv_hess(cone::EpiPerSquare)
    @assert cone.is_feas
    mul!(cone.inv_hess.data, cone.point, cone.point')
    for j in 3:cone.dim
        cone.inv_hess.data[j, j] += cone.dist
    end
    cone.inv_hess.data[1, 2] -= cone.dist
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSquare)
    @assert cone.is_feas
    for j in 1:size(prod, 2)
        pa = dot(cone.point, view(arr, :, j)) / cone.dist
        @. prod[:, j] = pa * cone.point
        @views @. prod[3:end, j] += arr[3:end, j]
        prod[1, j] -= arr[2, j]
        prod[2, j] -= arr[1, j]
    end
    prod ./= cone.dist
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSquare)
    @assert cone.is_feas
    for j in 1:size(prod, 2)
        pa = dot(cone.point, view(arr, :, j))
        @. prod[:, j] = pa * cone.point
        @views @. prod[3:end, j] += cone.dist * arr[3:end, j]
        prod[1, j] -= cone.dist * arr[2, j]
        prod[2, j] -= cone.dist * arr[1, j]
    end
    return prod
end



# TODO? hess_prod! and inv_hess_prod!


#
# u = cone.point[1]
# v = cone.point[2]
# w = view(cone.point, 3:cone.dim)
#
# @. cone.g = cone.point / dist
# tmp = -cone.g[2]
# cone.g[2] = -cone.g[1]
# cone.g[1] = tmp
#
# Hi = cone.Hi
# mul!(Hi, cone.point, cone.point') # TODO syrk
# Hi[2, 1] = Hi[1, 2] = u * v - cone.dist # TODO only need upper tri
# for j in 3:cone.dim
#     Hi[j, j] += dist
# end
#
# H = cone.H
# @. H = Hi
# for j in 3:cone.dim
#     H[1, j] = H[j, 1] = -Hi[2, j]
#     H[2, j] = H[j, 2] = -Hi[1, j]
# end
# H[1, 1] = Hi[2, 2]
# H[2, 2] = Hi[1, 1]
# @. H = H / dist / dist


# calcg!(g::AbstractVector{T}, cone::EpiPerSquare) = (@. g = cone.point/cone.dist; tmp = g[1]; g[1] = -g[2]; g[2] = -tmp; g)
# calcHiarr!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare) = mul!(prod, cone.Hi, arr)
# calcHarr!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare) = mul!(prod, cone.H, arr)

# inv_hess(cone::EpiPerSquare) = Symmetric(cone.Hi, :U)
#
# inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare{T}) where {T <: HypReal} = mul!(prod, Symmetric(cone.Hi, :U), arr)
