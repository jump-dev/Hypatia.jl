#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of matrix spectral norm (operator norm associated with standard Euclidean norm; i.e. maximum singular value)
(u in R, W in R^{n,m}) : u >= opnorm(W)
note n <= m is enforced WLOG since opnorm(W) = opnorm(W')
W is vectorized column-by-column (i.e. vec(W) in Julia)

barrier from "Interior-Point Polynomial Algorithms in Convex Programming" by Nesterov & Nemirovskii 1994
-logdet(u*I_n - W*W'/u) - log(u)

TODO eliminate allocations
=#

mutable struct EpiNormSpectral{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    n::Int
    m::Int
    point::AbstractVector{T}

    is_feas::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    W::Matrix{T}
    F

    function EpiNormSpectral{T}(n::Int, m::Int, is_dual::Bool) where {T <: HypReal}
        @assert n <= m
        dim = n * m + 1
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.n = n
        cone.m = m
        return cone
    end
end

EpiNormSpectral{T}(n::Int, m::Int) where {T <: HypReal} = EpiNormSpectral{T}(n, m, false)

function setup_data(cone::EpiNormSpectral{T}) where {T <: HypReal}
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.W = Matrix{T}(undef, cone.n, cone.m)
    return
end

get_nu(cone::EpiNormSpectral) = cone.n + 1

function set_initial_point(arr::AbstractVector, cone::EpiNormSpectral)
    arr .= 0
    arr[1] = 1
    return arr
end

reset_data(cone::EpiNormSpectral) = (cone.is_feas = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function update_feas(cone::EpiNormSpectral)
    @assert !cone.is_feas
    u = cone.point[1]
    if u > 0
        cone.W .= view(cone.point, 2:cone.dim)
        X = Symmetric(cone.W * cone.W') # TODO use syrk
        Z = Symmetric(u * I - X / u)
        fact_Z = hyp_chol!(Z)
        cone.is_feas = isposdef(fact_Z)
    end
    return cone.is_feas
end

function update_grad(cone::EpiNormSpectral)
    @assert cone.is_feas
    Zi = Symmetric(inv(cone.fact_Z))
    Eu = Symmetric(I + X / u / u)
    cone.g[1] = -dot(Zi, Eu) - inv(u)
    cone.g[2:end] .= vec(2 * Zi * W / u)
    cone.grad_updated = true
    return cone.grad
end

# TODO maybe this could be simpler/faster if use mul!(cone.hess, cone.grad, cone.grad') and build from there
# TODO remove allocations
function update_hess(cone::EpiNormSpectral)
    @assert cone.grad_updated
    cone.hess .= 0

    ZiEuZi = Symmetric(Zi * Eu * Zi)
    cone.H[1, 1] = dot(ZiEuZi, Eu) + (2 * dot(Zi, X) / u + one(T)) / u / u
    cone.H[1, 2:end] = vec(-2 * (ZiEuZi + Zi / u) *  W / u)

    p = 2
    for j in 1:m, i in 1:n
        tmpmat = Zi[:, i] * W[:, j]' * Zi / u

        # Zi * dZdWij * Zi
        term1 = Symmetric(tmpmat + tmpmat') # TODO use syrk

        q = p
        cone.H[p, q:(q + n - i)] = Zi[i, i:n]
        for ni in 1:n
            cone.H[p, q:(q + n - i)] += term1[ni, i:n] * W[ni, j]
        end
        cone.H[p, q:(q + n - i)] *= 2 / u
        q += (n - i + 1)

        mat = 2 * term1 * W[:, (j + 1):m] / u
        nterms = n * (m - j)
        cone.H[p, q:(q + nterms - 1)] += vec(mat)
        q += nterms
        p += 1
    end

    cone.hess_updated = true
    return cone.hess
end

# TODO? hess_prod! and inv_hess_prod!


# W = cone.W
# W[:] = view(cone.point, 2:cone.dim)
# n = cone.n
# m = cone.m
#
# X = Symmetric(W * W') # TODO use syrk
# Z = Symmetric(u * I - X / u)
# F = hyp_chol!(Z)
# if !isposdef(F)
#     return false
# end
#
# # TODO figure out structured form of inverse? could simplify algebra
# Zi = Symmetric(inv(F))
# Eu = Symmetric(I + X / u / u)
# cone.H .= zero(T)
#
# cone.g[1] = -dot(Zi, Eu) - inv(u)
# cone.g[2:end] = vec(2 * Zi * W / u)
#
# ZiEuZi = Symmetric(Zi * Eu * Zi)
# cone.H[1, 1] = dot(ZiEuZi, Eu) + (2 * dot(Zi, X) / u + one(T)) / u / u
# cone.H[1, 2:end] = vec(-2 * (ZiEuZi + Zi / u) *  W / u)
#
# p = 2
# for j in 1:m, i in 1:n
#     tmpmat = Zi[:, i] * W[:, j]' * Zi / u
#
#     # Zi * dZdWij * Zi
#     term1 = Symmetric(tmpmat + tmpmat') # TODO use syrk
#
#     q = p
#     cone.H[p, q:(q + n - i)] = Zi[i, i:n]
#     for ni in 1:n
#         cone.H[p, q:(q + n - i)] += term1[ni, i:n] * W[ni, j]
#     end
#     cone.H[p, q:(q + n - i)] *= 2 / u
#     q += (n - i + 1)
#
#     mat = 2 * term1 * W[:, (j + 1):m] / u
#     nterms = n * (m - j)
#     cone.H[p, q:(q + nterms - 1)] += vec(mat)
#     q += nterms
#     p += 1
# end
