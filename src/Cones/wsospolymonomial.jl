#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

using DynamicPolynomials, ForwardDiff, DiffResults

mutable struct WSOSPolyMonomial{T <: Real} <: Cone{T}
    use_dual::Bool
    n::Int
    deg::Int
    dim::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    poly_pairs
    barfun
    diffres

    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact
    hess_fact_cache

    function WSOSPolyMonomial{T}(n::Int, deg::Int, is_dual::Bool) where {T <: Real}
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.n = n
        cone.deg = deg
        dim = binomial(cone.n + deg, cone.n)
        cone.dim = dim
        return cone
    end
end

WSOSPolyMonomial{T}(n::Int, deg::Int) where {T <: Real} = WSOSPolyMonomial{T}(n::Int, deg::Int, false)

function setup_data(cone::WSOSPolyMonomial{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)

    # create a lookup table of coefficients we will use to build lambda later
    @polyvar x[1:cone.n]
    monos_low = monomials(x, 0:div(cone.deg, 2))
    monos_sqr = monomials(x, 0:cone.deg)
    L = length(monos_low)
    poly_pairs = [Float64[] for i in 1:L, j in 1:L]
    for i in 1:length(monos_low), j in 1:i, m in monos_sqr
        poly = monos_low[i] * monos_low[j]
        push!(poly_pairs[i, j], coefficient(poly, m))
    end
    cone.poly_pairs = poly_pairs

    function barfun(point)
        lambda = zeros(eltype(point), L, L)
        for k in 1:L, l in 1:k
            lambda[k, l] = dot(poly_pairs[k, l], point)
        end
        return -logdet(Symmetric(lambda, :L))
    end
    cone.barfun = barfun
    cone.diffres = DiffResults.HessianResult(cone.grad)

    cone.hess_fact_cache = nothing
    return
end

get_nu(cone::WSOSPolyMonomial) = binomial(cone.n + div(cone.deg, 2), cone.n)

function set_initial_point(arr::AbstractVector, cone::WSOSPolyMonomial)
    @polyvar x[1:cone.n]
    monos = monomials(x, 0:cone.deg)

    pts = randn(cone.dim, cone.n)
    V = zeros(cone.dim, cone.dim)
    for i in 1:cone.dim, j in 1:cone.dim
        V[i, j] = monos[i](pts[j, :])
    end
    arr .= V * ones(cone.dim)
    println(arr)

    # for (i, m) in enumerate(monos)
    #     if !all(iseven, exponents(m))
    #         arr[i] = 0
    #         continue
    #     end
    #     # arr[i] = inv(degree(m) + 1)
    #     arr[i] = 1
    #     # if degree(m) == 0
    #     #     arr[i] = 1
    #     # else
    #     #     arr[i] = inv(prod((iszero(e) ? 1 : e) for e in exponents(m)) + 1)
    #     # end
    #     # @show arr[i], inv(degree(m) + 1)
    # end
    # for i in eachindex(arr)
    #     arr[i] = inv(i + 1)
    # end
end

function update_feas(cone::WSOSPolyMonomial)
    @assert !cone.feas_updated
    L = binomial(cone.n + div(cone.deg, 2), cone.n)
    lambda = zeros(eltype(cone.point), L, L)
    for k in 1:L, l in 1:k
        lambda[k, l] = dot(cone.poly_pairs[k, l], cone.point)
    end
    @show Symmetric(lambda, :L)
    cone.is_feas = isposdef(Symmetric(lambda, :L))
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSPolyMonomial)
    @assert cone.is_feas
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.grad .= DiffResults.gradient(cone.diffres)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSPolyMonomial)
    @assert cone.grad_updated
    cone.hess.data .= DiffResults.hessian(cone.diffres)
    cone.hess_updated = true
    return cone.hess
end

# function inv_hess(cone::WSOSPolyMonomial)
#     cone.inv_hess = inv(cone.hess)
#     cone.inv_hess_updated = true
#     return cone.inv_hess
# end
