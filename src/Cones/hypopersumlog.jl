#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^n) : u <= v*sum(log.(w/v))

barrier (guessed, reduces to 3-dim exp cone self-concordant barrier)
-log(v*sum(log.(w/v)) - u) - sum(log.(w)) - log(v)

TODO
- rename to and replace 3D cone (hypoperlog)
- remove vw and other numerical optimization
=#

mutable struct HypoPerSumLog{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    k::Float64
    gamma::Float64

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    F

    function HypoPerSumLog{T}(dim::Int, is_dual::Bool; alpha = -2) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        if alpha == -1
            alpha = 4 + 3 / sqrt(cone.dim - 1)
        elseif alpha == -2
            cone.k = cone.dim
            cone.gamma = 1.0
            return cone
        end
        n = cone.dim - 1
        k = alpha * n
        cone.k = cone.dim + 2 # k
        cone.gamma = (k^(3 / 2) / (k - n)^(3 / 2) + (1 + k / (k - n))^(3 / 2) / sqrt(k))^2
        return cone
    end
end

HypoPerSumLog{T}(dim::Int; alpha = -2) where {T <: HypReal} = HypoPerSumLog{T}(dim, false, alpha = alpha)

function setup_data(cone::HypoPerSumLog{T}) where {T <: HypReal}
    dim = cone.dim
    cone.g = zeros(T, dim)
    cone.H = zeros(T, dim, dim)
    cone.H2 = copy(cone.H)
    function barfun(point)
        u = point[1]
        v = point[2]
        w = view(point, 3:dim)
        return cone.gamma * (-log(v * sum(wi -> log(wi / v), w) - u) - sum(wi -> log(wi), w) - log(v) * (cone.k - cone.dim + 1))
    end
    return
end

get_nu(cone::HypoPerSumLog) = cone.k * cone.gamma

function set_initial_point(arr::AbstractVector{T}, cone::HypoPerSumLog{T}) where {T <: HypReal}
    arr[1] = -one(T)
    @. arr[2:end] = one(T)
    return arr
end

# function check_in_cone(cone::HypoPerSumLog{T}) where {T <: HypReal}
#     u = cone.point[1]
#     v = cone.point[2]
#     w = view(cone.point, 3:cone.dim)
#     if any(wi -> wi <= zero(T), w) || v <= zero(T) || u >= v * sum(wi -> log(wi / v), w)
#         return false
#     end
#
#     cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
#     cone.g .= DiffResults.gradient(cone.diffres)
#     cone.H .= DiffResults.hessian(cone.diffres)
#
#     return factorize_hess(cone)
# end

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
    g[2] = (cone.dim - 2 - lwv) * ivlwvu - 1 / v * (cone.k - cone.dim + 1)
    g[3:end] = -(one(T) + v * ivlwvu) ./ w
    cone.g .*= cone.gamma

    # Hessian
    vw = v ./ w
    ivlwvu2 = abs2(ivlwvu)
    H = cone.H
    H[1, 1] = ivlwvu2
    H[1, 2] = H[2, 1] = -(lwv - one(T) * n) * ivlwvu2
    H[1, 3:end] = H[3:end, 1] = -vw * ivlwvu2
    H[2, 2] = abs2(lwv - one(T) * n) * ivlwvu2 + ivlwvu * one(T) * n / v + inv(abs2(v)) * (cone.k - cone.dim + 1)
    H[2, 3:end] = H[3:end, 2] = vw * (lwv - one(T) * n) * ivlwvu2 - ivlwvu ./ w
    for i in 1:(cone.dim - 2)
        for j in 1:(i - 1)
            H[(2 + i), (2 + j)] = H[(2 + j), (2 + i)] = ivlwvu2 * vw[i] * vw[j]
        end
        H[(2 + i), (2 + i)] = abs2(vw[i]) * ivlwvu2 + vw[i] / w[i] * ivlwvu + inv(abs2(w[i]))
    end
    cone.H .*= cone.gamma

    return factorize_hess(cone)
end
