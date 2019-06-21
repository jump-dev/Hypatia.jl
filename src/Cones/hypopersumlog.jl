#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^n) : u <= v*sum(log.(w/v))

barrier (guessed, reduces to 3-dim exp cone self-concordant barrier)
-log(v*sum(log.(w/v)) - u) - sum(log.(w)) - log(v)

TODO
- rename to and replace 3D cone (hypoperlog)
=#

mutable struct HypoPerSumLog{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    F

    function HypoPerSumLog{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

HypoPerSumLog{T}(dim::Int) where {T <: HypReal} = HypoPerSumLog{T}(dim, false)

function setup_data(cone::HypoPerSumLog{T}) where {T <: HypReal}
    dim = cone.dim
    cone.g = zeros(T, dim)
    cone.H = zeros(T, dim, dim)
    cone.H2 = copy(cone.H)
    function barfun(point)
        u = point[1]
        v = point[2]
        w = view(point, 3:dim)
        return -log(v * sum(wi -> log(wi / v), w) - u) - sum(wi -> log(wi), w) - log(v)
    end
    return
end

get_nu(cone::HypoPerSumLog) = cone.dim

function set_initial_point(arr::AbstractVector{T}, cone::HypoPerSumLog{T}) where {T <: HypReal}
    arr[1] = -one(T)
    arr[2] = one(T)
    @. arr[3:end] = one(T)
    return arr
end

function check_in_cone(cone::HypoPerSumLog{T}) where {T <: HypReal}
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    if any(wi -> wi <= zero(T), w) || v <= zero(T) || u >= v * sum(wi -> log(wi / v), w)
        return false
    end

    lwv = sum(wi -> log(wi / v), w)
    vlwv = v * lwv
    vlwvu = vlwv - u
    n = cone.dim - 2

    # gradient
    ivlwvu = inv(vlwvu)
    g = cone.g
    g[1] = ivlwvu
    g[2] = (cone.dim - 2 - lwv) * ivlwvu - 1 / v
    g[3:end] = -(one(T) + v * ivlwvu) ./ w

    # Hessian
    vw = v ./ w # TODO remove allocations
    ivlwvu2 = abs2(ivlwvu)
    H = cone.H
    H[1, 1] = ivlwvu2
    H[1, 2] = -(lwv - T(n)) * ivlwvu2
    H[1, 3:end] = -vw * ivlwvu2
    H[2, 2] = abs2(lwv - T(n)) * ivlwvu2 + ivlwvu * T(n) / v + inv(abs2(v))
    H[2, 3:end] = vw * (lwv - T(n)) * ivlwvu2 - ivlwvu ./ w
    for j in 1:(cone.dim - 2)
        for i in 1:(j - 1)
            H[(2 + i), (2 + j)] = ivlwvu2 * vw[i] * vw[j]
        end
        H[(2 + j), (2 + j)] = abs2(vw[j]) * ivlwvu2 + vw[j] / w[j] * ivlwvu + inv(abs2(w[j]))
    end

    return factorize_hess(cone)
end
