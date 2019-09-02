#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

using DynamicPolynomials, ForwardDiff, DiffResults

mutable struct SOSConvexPolyMonomial{T <: Real} <: Cone{T}
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

    function SOSConvexPolyMonomial{T}(n::Int, deg::Int, is_dual::Bool) where {T <: Real}
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.n = n
        cone.deg = deg
        dim = binomial(cone.n + deg, cone.n)
        cone.dim = dim
        return cone
    end
end

SOSConvexPolyMonomial{T}(n::Int, deg::Int) where {T <: Real} = SOSConvexPolyMonomial{T}(n::Int, deg::Int, false)

function setup_data(cone::SOSConvexPolyMonomial{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)

    # create a lookup table of coefficients we will use to build lambda later
    n = cone.n
    @polyvar x[1:n]
    monos_sqr = monomials(x, 2:cone.deg)
    monos_hess = monomials(x, 0:cone.deg - 2)
    monos_low = monomials(x, 0:div(cone.deg - 2, 2))
    L = length(monos_low)
    poly_pairs = [Float64[] for i in 1:L, j in 1:L]
    for i in 1:length(monos_low), j in 1:i, m in monos_hess
        poly = monos_low[i] * monos_low[j]
        push!(poly_pairs[i, j], coefficient(poly, m))
    end
    cone.poly_pairs = poly_pairs

    lifting = zeros(length(monos_hess) * div(n * (n + 1), 2), length(monos_sqr))
    for k in 1:length(monos_sqr)
        basis_poly = monos_sqr[k]
        hess = differentiate(basis_poly, x, 2)
        lifting[:, k] = vcat([coefficient(hess[i, j], m) for i in 1:n for j in 1:i for m in monos_hess]...)
    end
    integrating = similar(lifting)
    for i in 1:size(lifting, 1), j in 1:size(lifting, 2)
        aij = lifting[i, j]
        integrating[i, j] = iszero(aij) ? 0 : inv(aij)
    end
    scalevals = diag(lifting' * integrating)
    integrating ./= scalevals'

    function barfun(point)
        hess_fullspace = integrating * point
        lambda = zeros(eltype(point), n * L, n * L)
        u = 1
        for i in 1:n, j in 1:i
            point_coeffs = view(point, u:(u + U - 1))
            for k in 1:L, l in 1:k
                lambda[(i - 1) * L + k, (j - 1) * L + l] = lambda[(i - 1) * L + l, (j - 1) * L + k] = dot(poly_pairs[k, l], point_coeffs)
            end
            u += U
        end
        return -logdet(Symmetric(lambda, :L))
    end
    cone.barfun = barfun
    cone.diffres = DiffResults.HessianResult(cone.grad)

    cone.hess_fact_cache = nothing
    return
end

get_nu(cone::SOSConvexPolyMonomial) = binomial(cone.n + div(cone.deg, 2), cone.n)

function set_initial_point(arr::AbstractVector, cone::SOSConvexPolyMonomial)
    arr .= 0
    arr[1] = 1
    arr[3] = 1
    arr[5] = 2
    return arr
end

function update_feas(cone::SOSConvexPolyMonomial)
    @assert !cone.feas_updated
    L = binomial(cone.n + div(cone.deg, 2), cone.n)
    lambda = zeros(eltype(cone.point), L, L)
    for k in 1:L, l in 1:k
        lambda[k, l] = dot(cone.poly_pairs[k, l], cone.point)
    end
    cone.is_feas = isposdef(Symmetric(lambda, :L))
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::SOSConvexPolyMonomial)
    @assert cone.is_feas
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.grad .= DiffResults.gradient(cone.diffres)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::SOSConvexPolyMonomial)
    @assert cone.grad_updated
    cone.hess.data .= DiffResults.hessian(cone.diffres)
    cone.hess_updated = true
    return cone.hess
end

# function inv_hess(cone::SOSConvexPolyMonomial)
#     cone.inv_hess = inv(cone.hess)
#     cone.inv_hess_updated = true
#     return cone.inv_hess
# end
