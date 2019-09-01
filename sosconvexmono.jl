#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

using ForwardDiff, DynamicPolynomials, LinearAlgebra

n = 2
k = 4
k_l = div(k - 2, 2)

@polyvar x[1:n]
monos_sqr = monomials(x, 2:k)
monos_hess = monomials(x, 0:(k - 2))
monos_low = monomials(x, 0:k_l)

# lookup table for when we come to calculate lambda, stores coefficients of p_i * p_j
# and lambda_{ij}(x) = dot(coefficients(p_i * p_j), x) = dot(poly_pairs[i, j], x)
poly_pairs = [Float64[] for i in 1:length(monos_low), j in 1:length(monos_low)]
for i in 1:length(monos_low), j in 1:i, m in monos_hess
    poly = monos_low[i] * monos_low[j]
    push!(poly_pairs[i, j], coefficient(poly, m))
end

# same thing as what we do in the monomial SOS cone, in block form
function monomial_lambda(point)
    L = binomial(n + k_l, n)
    @assert L == length(monos_low)
    U = binomial(n + k - 2, n)
    @assert U == length(monos_hess)

    lambda = zeros(eltype(point), n * L, n * L)
    u = 1
    for i in 1:n, j in 1:i
        point_coeffs = view(point, u:(u + U - 1))
        for k in 1:L, l in 1:k
            lambda[(i - 1) * L + k, (j - 1) * L + l] = lambda[(i - 1) * L + l, (j - 1) * L + k] = dot(poly_pairs[k, l], point_coeffs)
        end
        u += U
    end
    return lambda
end

# # (old code) this is how we could get Lambda using Dynamic polynomials
# function get_lambda_dp(point)
#     p = dot(point, monos_sqr)
#     hess = differentiate(p, x, 2)
#     hess_fullspace = vcat([coefficient(hess[i, j], m) for i in 1:n for j in 1:i for m in monos_hess]...)
#     lambda = monomial_lambda(hess_fullspace)
#     return lambda
# end

# but we don't want to use dynamic polynomials (to be able to use forwarddiff, and also get the lifting matrix explicitly for interest)
# the lifting takes us from a polynomial up to degree k to a vcat-ed hessian with polys of maximum degree k - 2
lifting = zeros(length(monos_hess) * div(n * (n + 1), 2), length(monos_sqr))
# the lifting is equal to the concatenation of its action basis vectors
for k in 1:length(monos_sqr)
    basis_poly = monos_sqr[k]
    hess = differentiate(basis_poly, x, 2)
    lifting[:, k] = vcat([coefficient(hess[i, j], m) for i in 1:n for j in 1:i for m in monos_hess]...)
end
# get lambda without using dynamic polynomials
function get_lambda(point)
    # hess_fullspace = lifting * point
    hess_fullspace = integrate(lifting) * point
    lambda = monomial_lambda(hess_fullspace)
    return lambda
end
# check lifting matrix is correct
point = randn(length(monos_sqr))
# @assert get_lambda(point) ≈ get_lambda_dp(point)

# barrier function
function barfun(point)
    lambda = get_lambda(point)
    f = cholesky(Symmetric(lambda, :L), check = false)
    @show eigmin(Symmetric(lambda, :L))
    @assert isposdef(f)
    return -logdet(f)
end

function feas_check(point)
    lambda = get_lambda(point)
    @show lambda
    f = cholesky(Symmetric(lambda, :L), check = false)
    @show eigmin(Symmetric(lambda, :L))
    @show isposdef(f)
    return isposdef(f)
end

function integrate(lifting)
    integrating = similar(lifting)
    for i in 1:size(lifting, 1), j in 1:size(lifting, 2)
        aij = lifting[i, j]
        integrating[i, j] = iszero(aij) ? 0 : inv(aij)
    end
    scalevals = diag(lifting' * integrating)
    integrating ./= scalevals'
    @show lifting' * integrating
    return integrating
end


for (i, m) in enumerate(monos_sqr)
    point[i] = all(iseven, exponents(m)) ? 1 : 0
end

feas_check(point)

# gradient = ForwardDiff.gradient(barfun, point)
# @show dot(-gradient, point)
# hessian = ForwardDiff.hessian(barfun, point)
# @assert isposdef(Symmetric(hessian))


# for _ in 1:100
#
#     b = rand()
#     c = rand()
#     a = 36 * b ^ 2 / 96 / c * 1.2
#     @assert 36 * b ^ 2 - 96 * a * c < 0
#
#     u = rand()
#     v = rand()
#     w = v ^ 2 / u * 1.1
#     # w = 2 / 3 * v ^ 2 / u * 1.1
#
#     @assert dot([a, b, c], [u, v, w]) > 0
#
#     # point = [rand(), 0, rand()]
#     point = [w, v, u]
#     @show point
#     feas_check(point)
#
#     # @show (u * a + v * b + w * c < 0) && !feas_check(point)
#     # @assert ((u * a + v * b + w * c > 0) && feas_check(point)) || ((u * a + v * b + w * c < 0) && !feas_check(point))
#     println()
# end


# using SumOfSquares, Hypatia
# model = SOSModel(with_optimizer(Mosek.Optimizer))
# @variable(model, poly, Poly(monos_sqr))
# hess = differentiate(poly, x, 2)
# @constraint(model, poly in SOSConvexCone())
# @constraint(model, sum(coefficients(poly)) == 1)
# # optimize!(model)
# # println(coefficients(JuMP.value.(poly)))
# @variable(model, t)
# @objective(model, Max, t)
# @constraint(model, hess - t * I in SOSMatrixCone())
# optimize!(model)
# println(coefficients(JuMP.value.(poly)))
# point = coefficients(JuMP.value.(poly))


# k = 6
# n = 3
# @polyvar x[1:n]
# monos = monomials(x, 0:3)
# m = SOSModel(with_optimizer(Mosek.Optimizer))
# @variables(m, p, Poly(monomials(x, 2:k)))
# @variable(m, G[1:length(monos), 1:length(monos)], Symmetric)
# @variable(m, t)
# @constraint(m, constr, p in SOSCone())
# @constraint(m, p == monos' * G * monos)
# @constraint(m, tr(G) == 1)
# @constraint(m, vcat(t, 1, [G[i, j] for i in 1:length(monos) for j in 1:i]...) in MOI.LogDetConeTriangle(length(monos)))
# @objective(m, Max, t)
# optimize!(m)
# @show value.(p)
# @show termination_status(m)
#
# k = 6
# n = 3
# @polyvar x[1:n]
# m = SOSModel(with_optimizer(Mosek.Optimizer))
# @variable(m, p, Poly(monomials(x, 2:k)))
# @constraint(m, constr, p in SOSCone())
# optimize!(m)
# @show value.(p)
# @show termination_status(m)

;
