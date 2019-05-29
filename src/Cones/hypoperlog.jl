#=
Copyright 2018, Chris Coey and contributors

(closure of) hypograph of perspective of (natural) log (AKA exponential cone)
(u in R, v in R_+, w in R_+) : u <= v*log(w/v)

barrier from "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization" by Skajaa & Ye 2014
-log(v*log(w/v) - u) - log(w) - log(v)

TODO allow different log bases?
TODO maybe use StaticArrays
TODO try to extend to case w in R^n
TODO could write the inverse hessian analytically rather than factorizing
TODO choose a better interior direction
=#

mutable struct HypoPerLog{T <: HypReal} <: Cone{T}
    use_dual::Bool

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    F

    function HypoPerLog{T}(is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        return cone
    end
end

HypoPerLog{T}() where {T <: HypReal} = HypoPerLog{T}(false)

function setup_data(cone::HypoPerLog{T}) where {T <: HypReal}
    cone.g = Vector{T}(undef, 3)
    cone.H = Matrix{T}(undef, 3, 3)
    cone.H2 = similar(cone.H)
    return
end

dimension(cone::HypoPerLog) = 3

get_nu(cone::HypoPerLog) = 3

set_initial_point(arr::AbstractVector{T}, cone::HypoPerLog{T}) where {T <: HypReal} = (arr[1] = -one(T); arr[2] = one(T); arr[3] = one(T); arr)

function check_in_cone(cone::HypoPerLog{T}) where {T <: HypReal}
    (u, v, w) = cone.point
    if (v <= zero(T)) || (w <= zero(T))
        return false
    end
    lwv = log(w / v)
    vlwv = v * lwv
    vlwvu = vlwv - u
    if vlwvu <= zero(T)
        return false
    end

    # gradient
    ivlwvu = inv(vlwvu)
    g = cone.g
    g[1] = ivlwvu
    g[2] = ivlwvu * (v - u - vlwvu - vlwvu) / v
    g[3] = -(one(T) + v * ivlwvu) / w

    # Hessian
    vw = v / w
    ivlwvu2 = abs2(ivlwvu)
    H = cone.H
    H[1, 1] = ivlwvu2
    H[1, 2] = H[2, 1] = -(lwv - one(T)) * ivlwvu2
    H[1, 3] = H[3, 1] = -vw * ivlwvu2
    H[2, 2] = abs2(lwv - one(T)) * ivlwvu2 + ivlwvu / v + inv(abs2(v))
    H[2, 3] = H[3, 2] = vw * (lwv - one(T)) * ivlwvu2 - ivlwvu / w
    H[3, 3] = abs2(vw) * ivlwvu2 + vw / w * ivlwvu + inv(abs2(w))

    return factorize_hess(cone)
end
