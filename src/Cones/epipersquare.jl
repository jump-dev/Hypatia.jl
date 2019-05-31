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
    g::Vector{T}
    H::Matrix{T}
    Hi::Matrix{T}

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
    cone.g = zeros(T, dim)
    cone.H = zeros(T, dim, dim)
    cone.Hi = copy(cone.H)
    return
end

get_nu(cone::EpiPerSquare) = 2

set_initial_point(arr::AbstractVector{T}, cone::EpiPerSquare{T}) where {T <: HypReal} = (@. arr = zero(T); arr[1] = one(T); arr[2] = one(T); arr)

function check_in_cone(cone::EpiPerSquare{T}) where {T <: HypReal}
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    if u <= zero(T) || v <= zero(T)
        return false
    end
    nrm2 = T(0.5) * sum(abs2, w)
    dist = u * v - nrm2
    if dist <= zero(T)
        return false
    end

    @. cone.g = cone.point / dist
    tmp = -cone.g[2]
    cone.g[2] = -cone.g[1]
    cone.g[1] = tmp

    Hi = cone.Hi
    mul!(Hi, cone.point, cone.point') # TODO syrk
    Hi[2, 1] = Hi[1, 2] = nrm2 # TODO only need upper tri
    for j in 3:cone.dim
        Hi[j, j] += dist
    end

    H = cone.H
    @. H = Hi
    for j in 3:cone.dim
        H[1, j] = H[j, 1] = -Hi[2, j]
        H[2, j] = H[j, 2] = -Hi[1, j]
    end
    H[1, 1] = Hi[2, 2]
    H[2, 2] = Hi[1, 1]
    @. H = H / dist / dist

    return true
end

# calcg!(g::AbstractVector{T}, cone::EpiPerSquare) = (@. g = cone.point/cone.dist; tmp = g[1]; g[1] = -g[2]; g[2] = -tmp; g)
# calcHiarr!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare) = mul!(prod, cone.Hi, arr)
# calcHarr!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare) = mul!(prod, cone.H, arr)

inv_hess(cone::EpiPerSquare) = Symmetric(cone.Hi, :U)

inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare{T}) where {T <: HypReal} = mul!(prod, Symmetric(cone.Hi, :U), arr)
