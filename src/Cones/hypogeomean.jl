#=
Copyright 2018, Chris Coey and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

dual barrier (modified by reflecting around u = 0 and using dual cone definition) from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((w_i/alpha_i)^alpha_i) + u) - sum_i((1 - alpha_i)*log(w_i/alpha_i)) - log(-u)

TODO try to make barrier evaluation more efficient
=#

mutable struct HypoGeomean{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    alpha::Vector{<:Real}

    ialpha::Vector{<:Real}
    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    F

    function HypoGeomean{T}(alpha::Vector{<:Real}, is_dual::Bool) where {T <: HypReal}
        dim = length(alpha) + 1
        @assert dim >= 3
        @assert all(ai >= 0.0 for ai in alpha)
        tol = 1e3 * eps(T)
        @assert sum(alpha) ≈ 1 atol=tol rtol=tol
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.alpha = alpha
        return cone
    end
end

HypoGeomean{T}(alpha::Vector{<:Real}) where {T <: HypReal} = HypoGeomean{T}(alpha, false)

function setup_data(cone::HypoGeomean{T}) where {T <: HypReal}
    dim = cone.dim
    alpha = cone.alpha
    ialpha = inv.(alpha)
    cone.ialpha = ialpha
    cone.g = zeros(T, dim)
    cone.H = zeros(T, dim, dim)
    cone.H2 = copy(cone.H)
    return
end

get_nu(cone::HypoGeomean) = cone.dim

set_initial_point(arr::AbstractVector{T}, cone::HypoGeomean{T}) where {T <: HypReal} = (@. arr = one(T); arr[1] = -prod(cone.ialpha[i]^cone.alpha[i] for i in eachindex(cone.alpha)) / cone.dim; arr)

function check_in_cone(cone::HypoGeomean{T}) where {T <: HypReal}
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    if u >= zero(T) || any(wi <= zero(T) for wi in w)
        return false
    end
    if sum(cone.alpha[i] * log(w[i] * cone.ialpha[i]) for i in eachindex(cone.alpha)) <= log(-u)
        return false
    end

    ialpha = cone.ialpha
    alpha = cone.alpha

    prodwaa = prod((w[i] * ialpha[i])^alpha[i] for i in eachindex(alpha))
    prodwaau = prodwaa + u

    cone.g[1] = -inv(prodwaau) - inv(u)
    cone.H[1, 1] = inv(abs2(prodwaau)) + inv(u) / u
    # column loop
    for i in eachindex(alpha)
        prod_excli = prodwaa * alpha[i] / w[i]
        cone.g[i + 1] = -prod_excli / prodwaau - (1 - alpha[i]) / w[i]
        cone.H[1, i + 1] = prod_excli / prodwaau / prodwaau
        fact = prodwaa / prodwaau * alpha[i] / w[i]
        # row loop
        for j in 1:(i - 1)
            prod_exclj = prodwaa * alpha[j] / w[j]
            cone.H[j + 1, i + 1] = fact * alpha[j] / w[j] * (prodwaa / prodwaau - 1)
        end
        cone.H[i + 1, i + 1] = fact * (prodwaa / prodwaau * alpha[i] - alpha[i] + 1) + (1 - alpha[i]) / w[i] / w[i]
    end

    return factorize_hess(cone)
end
