#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

using ForwardDiff, DynamicPolynomials, LinearAlgebra

n = 1
k = 6
k_l = div(k - 2, 2)

@polyvar x[1:n]
monos_sqr = monomials(x, 2:k)
monos_hess = monomials(x, 0:(k - 2))
monos_low = monomials(x, 0:k_l)

# lookup table for when we come to calculate lambda, stores coefficients of p_i * p_j
poly_pairs = [Float64[] for i in 1:length(monos_low), j in 1:length(monos_low)]
for i in 1:length(monos_low), j in 1:i, m in monos_hess
    poly = monos_low[i] * monos_low[j]
    push!(poly_pairs[i, j], coefficient(poly, m))
end

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
            fact = (i == j ? 1 : inv(sqrt(2)))
            lambda[(i - 1) * L + k, (j - 1) * L + l] = lambda[(i - 1) * L + l, (j - 1) * L + k] = dot(poly_pairs[k, l], point_coeffs) * fact
        end
        u += U
    end
    return lambda
end

# for interest (and to use forwarddiff), get the overconstrained LHS we are implicitly using to lift
lifting = zeros(length(monos_hess) * div(n * (n + 1), 2), length(monos_sqr))
# the lifting is equal to the concatenation of its action on each basis vector
for k in 1:length(monos_sqr)
    basis_poly = monos_sqr[k]
    hess = differentiate(basis_poly, x, 2)
    lifting[:, k] = vcat([coefficient(hess[i, j], m) for i in 1:n for j in 1:i for m in monos_hess]...)
end

# using dynamic polynomials, check lambda logic is right
function get_lambda_dp(point)
    p = dot(point, monos_sqr)
    hess = differentiate(p, x, 2)
    hess_fullspace = vcat([coefficient(hess[i, j], m) for i in 1:n for j in 1:i for m in monos_hess]...)
    lambda = monomial_lambda(hess_fullspace)
    return lambda
end

# without using dynamic polynomials
function get_lambda(point)
    hess_fullspace = lifting * point
    lambda = monomial_lambda(hess_fullspace)
    return lambda
end
point = randn(length(monos_sqr))
@assert get_lambda(point) ≈ get_lambda_dp(point)

# we can come up with an interior point for the univariate case
feasible_hess = zeros(length(monos_hess) * div(n * (n + 1), 2))
idx = 1
for i in 1:n, j in 1:i, u in 1:length(monos_sqr)
    if i == j
        feasible_hess[idx] = inv(u + 1)
    end
    global idx += 1
end
# luckily this is a Hessian!
point = lifting \ feasible_hess
@assert norm(feasible_hess - lifting * point) ≈ 0
barfun(point) = -logdet(get_lambda(point))
gradient = ForwardDiff.gradient(barfun, point)
@show dot(-gradient, point)





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
