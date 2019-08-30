#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points Ps

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO
- perform loop for calculating g and H in parallel
- scale the interior direction
=#

using DynamicPolynomials, ForwardDiff, DiffResults

mutable struct WSOSPolyMonomial{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    Ps::Vector{Matrix{R}}
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

    polypairs
    barfun

    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact
    hess_fact_cache

    function WSOSPolyMonomial{T, R}(n::Int, deg::Int, is_dual::Bool) where {R <: RealOrComplex{T}} where {T <: Real}
        cone = new{T, R}()
        cone.use_dual = !is_dual # using dual barrier
        dim = binomial(n + deg, n)
        cone.dim = dim
        return cone
    end
end

WSOSPolyMonomial{T, R}(n::Int, deg::Int) where {R <: RealOrComplex{T}} where {T <: Real} = WSOSPolyMonomial{T, R}(n::Int, deg::Int, false)

function setup_data(cone::WSOSPolyMonomial{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)

    # create a lookup table of coefficients we will use to build lambda later
    @polyvar x[1:n]
    monos_low = monomials(x, 0:div(cone.deg, 2))
    L = length(monos_low)
    polypairs = [Float64[] for i in 1:L, j in 1:L]
    for i in 1:length(monos_low), j in 1:i, m in monos_hess
        push!(poly_pairs[i, j], coefficient(poly, m).constant)
    end
    cone.polypairs = polypairs

    function barfun(point)
        lambda = zeros(eltype(point), L, L)
        for k in 1:L, l in 1:k
            lambda[k, l] = dot(polypairs[k, l], point)
        end
        return -logdet(lambda)
    end
    cone.barfun = barfun

    cone.hess_fact_cache = nothing
    return
end

get_nu(cone::WSOSPolyMonomial) = binomial(n + div(deg, 2), n)

set_initial_point(arr::AbstractVector, cone::WSOSPolyMonomial) = (arr .= 1)

function update_feas(cone::WSOSPolyMonomial)
    @assert !cone.feas_updated
    L = binomial(cone.n + div(cone.deg, 2), cone.n)
    lambda = zeros(eltype(cone.point), L, L)
    for k in 1:L, l in 1:k
        lambda[k, l] = dot(cone.polypairs[k, l], cone.point)
    end
    return isposdef(Symmetric(lambda, :L))
end

function update_grad(cone::Cone)
    @assert cone.is_feas
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.grad .= DiffResults.gradient(cone.diffres)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Cone)
    @assert cone.grad_updated
    cone.hess.data .= DiffResults.hessian(cone.diffres)
    cone.hess_updated = true
    return cone.hess
end
