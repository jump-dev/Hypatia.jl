#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

using ForwardDiff, DynamicPolynomials, LinearAlgebra

n = 2
k = 6
k_l = div(k - 2, 2)

@polyvar x[1:n]

monos_sqr = monomials(x, 2:k)
monos_hess = monomials(x, 0:(k - 2))
monos_low = monomials(x, 0:k_l)

# point = randn(length(monos_sqr))

function get_sqr_coeffs(i, j)
    poly = monos_low[i] * monos_low[j]
    return [coefficient(poly, m) for m in monos_hess]
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
            lambda[(i - 1) * L + k, (j - 1) * L + l] = lambda[(i - 1) * L + l, (j - 1) * L + k] = dot(get_sqr_coeffs(k, l), point_coeffs)
        end
        u += U
    end
    return lambda
end

function feas_check(point)
    p = dot(point, monos_sqr)
    # using DP
    hess = differentiate(p, x, 2)
    hess_fullspace = vcat([coefficient(hess[i, j], m) for i in 1:n for j in 1:i for m in monos_hess]...)
    # without using DP
    # hess_fullspace = get_lifting() * point
    lambda = monomial_lambda(hess_fullspace)
    f = cholesky(Symmetric(lambda, :L), check = false)
    return f
end

# for interest, get the overconstrained LHS we are implicitly using to lift
function get_lifting()
    lifting = zeros(length(monos_hess) * div(n * (n + 1), 2), length(monos_sqr))
    for k in 1:length(monos_sqr)
        basis_poly = monos_sqr[k]
        hess = differentiate(basis_poly, x, 2)
        lifting[:, k] = vcat([coefficient(hess[i, j], m) for i in 1:n for j in 1:i for m in monos_hess]...)
    end
    return lifting
end

# check matrix with ones on the diagonal
# In the interpolant basis, we know that the measure "1" is in the interior and
# we use "1 on the diagonal" for the SOS-matrix cone. This does correspond to a
# "hessian" matrix, but not in the interior
one_moment = zeros(length(monos_hess))
one_moment[end] = 1
feasible_hess = zeros(length(monos_hess) * div(n * (n + 1), 2))
idx = 1
for i in 1:n, j in 1:i, u in 1:length(monos_hess)
    if i == j && u == length(monos_hess)
        feasible_hess[idx] = 1
    end
    global idx += 1
end
lifting = get_lifting()
point = lifting \ feasible_hess
@assert dot(point, monos_sqr) ≈ 0.5 * x[1] ^ 2 + 0.5 * x[2] ^ 2


barfun(point) = -logdet(feas_check(point))


# convex polys
p1 = x[1] ^ 6 + x[2] ^ 4
point = [coefficient(p1, m) for m in monos_sqr]
p2 = x[1] ^ 6 + x[1] ^ 4 + x[1] ^ 2 + x[2] ^ 2 + 1

# non-convex polys
q1 = x[1] ^ 6 * rand() + x[1] ^ 5 * rand() + x[2] ^ 6 * rand() + x[2] ^ 5


point = zeros(length(monos_sqr))
point[1] = 1
point[7] = 1

gradient = ForwardDiff.gradient(barfun, point)





;
