#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of matrix spectral norm (operator norm associated with standard Euclidean norm; i.e. maximum singular value)
(u in R, W in R^{n,m}) : u >= opnorm(W)
note n <= m is enforced WLOG since opnorm(W) = opnorm(W')
W is vectorized column-by-column (i.e. vec(W) in Julia)

barrier from "Interior-Point Polynomial Algorithms in Convex Programming" by Nesterov & Nemirovskii 1994
-logdet(u*I_n - W*W'/u) - log(u)

TODO eliminate allocations
TODO type auxiliary fields
=#

mutable struct EpiNormSpectral{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    n::Int
    m::Int
    point::AbstractVector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    W
    X
    fact_Z
    Zi
    Eu
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact # TODO prealloc

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
    reset_data(cone)
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.W = Matrix{T}(undef, cone.n, cone.m)
    return
end

get_nu(cone::EpiNormSpectral) = cone.n + 1

function set_initial_point(arr::AbstractVector, cone::EpiNormSpectral)
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::EpiNormSpectral)
    @assert !cone.feas_updated
    u = cone.point[1]
    if u > 0
        cone.W[:] .= view(cone.point, 2:cone.dim)
        cone.X = Symmetric(cone.W * cone.W') # TODO use syrk
        Z = Symmetric(u * I - cone.X / u)
        cone.fact_Z = hyp_chol!(Z)
        cone.is_feas = isposdef(cone.fact_Z)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormSpectral)
    @assert cone.is_feas
    u = cone.point[1]
    cone.Zi = Symmetric(inv(cone.fact_Z))
    cone.Eu = Symmetric(I + cone.X / u / u)
    cone.grad[1] = -dot(cone.Zi, cone.Eu) - inv(u)
    cone.grad[2:end] .= vec(2 * cone.Zi * cone.W / u)
    cone.grad_updated = true
    return cone.grad
end

# TODO maybe this could be simpler/faster if use mul!(cone.hess, cone.grad, cone.grad') and build from there
function update_hess(cone::EpiNormSpectral)
    @assert cone.grad_updated
    n = cone.n
    m = cone.m
    u = cone.point[1]
    W = cone.W
    X = cone.X
    Zi = cone.Zi
    Eu = cone.Eu
    cone.hess .= 0
    ZiEuZi = Symmetric(Zi * Eu * Zi)
    cone.hess.data[1, 1] = dot(ZiEuZi, Eu) + (2 * dot(Zi, X) / u + 1) / u / u
    cone.hess.data[1, 2:end] = vec(-2 * (ZiEuZi + Zi / u) * W / u)
    p = 2
    for j in 1:m, i in 1:n
        @views tmpmat = Zi[:, i] * W[:, j]' * Zi / u
        term1 = Symmetric(tmpmat + tmpmat') # Zi * dZdWij * Zi
        q = p
        viewij = view(cone.hess.data, p, q:(q + n - i))
        @views viewij .= Zi[i, i:n]
        @views @. for ni in 1:n
            viewij += W[ni, j] * term1[ni, i:n]
        end
        viewij .*= 2 / u
        q += (n - i + 1)
        ntermsij = n * (m - j)
        @views cone.hess.data[p, q:(q + ntermsij - 1)] .+= vec(2 * term1 * W[:, (j + 1):m] / u)
        q += ntermsij
        p += 1
    end
    cone.hess_updated = true
    return cone.hess
end
