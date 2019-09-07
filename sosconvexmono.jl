#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

using ForwardDiff, DynamicPolynomials, LinearAlgebra, JuMP, MosekTools, Hypatia

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
    # @show eigmin(Symmetric(lambda, :L))
    @assert isposdef(f)
    return -logdet(f)
end

function feas_check(point)
    lambda = get_lambda(point)
    @show lambda
    f = cholesky(Symmetric(lambda, :L), check = false)
    # @show eigmin(Symmetric(lambda, :L))
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
    # @show lifting' * integrating
    return integrating
end

# point = zeros(length(monos_sqr))
# point[1] = 12
# point[3] = 12
# point[5] = 1.01 # *
# point[10] = 1
# point[12] = 1
# h = lifting' \ point
#
# lambda = monomial_lambda(h)
# barfun(point)
# @show get_lambda(point)
# gradient = ForwardDiff.gradient(barfun, point)
# @show dot(-gradient, point)
# hessian = ForwardDiff.hessian(barfun, point)
# @assert isposdef(Symmetric(hessian))

function lambda_direct(point)
    L = length(monos_low)
    lambda = zeros(eltype(point), L * n, L * n)
    for (u, msu) in enumerate(monos_sqr)
        idxs = []
        num_repeats = 0
        for i in 1:n, j in 1:i
            di = degree(msu, x[i])
            dj = degree(msu, x[j])
            if (i == j && di >= 2) || (i != j && di >= 1 && dj >= 1)
                num_repeats += 1
            else
                continue
            end
            for k in 1:length(monos_low), l in 1:k
                if msu != monos_low[k] * monos_low[l] * x[i] * x[j]
                    continue
                end
                if i == j
                    fact = inv(di * (di - 1))
                else
                    fact = inv(di * dj)
                end
                lambda[(i - 1) * L + k, (j - 1) * L + l] = lambda[(i - 1) * L + l, (j - 1) * L + k] = fact * point[u]
                push!(idxs, (i, j, k, l))
            end # inner
        end # outer
        println("$msu has $num_repeats repeats")
        # if msu == x[1] ^ 2 * x[2] ^ 2
        #     weights = [0.5, 0, 0.5]
        # else
            weights = fill(1 / num_repeats, num_repeats)
        # end
        for (w, (i, j, k, l)) in enumerate(idxs)
            lambda[(i - 1) * L + k, (j - 1) * L + l] *= weights[w]
            if k != l
                lambda[(i - 1) * L + l, (j - 1) * L + k] *= weights[w]
            end
        end
    end # monos sqr
    return lambda
end


my_nullspace = zeros(18, 6)
my_nullspace[5, 1] = 1
my_nullspace[10, 1] = -1
my_nullspace[2, 2] = (1 / 6)
my_nullspace[7, 2] = (-1 / 3)
my_nullspace[16, 3] = (1 / 2)
my_nullspace[11, 3] = (-1 / 2)
my_nullspace[14, 4] = (1 / 6)
my_nullspace[9, 4] = (-1 / 3)
my_nullspace[3, 5] = (1 / 2)
my_nullspace[8, 5] = (-1 / 4)
my_nullspace[8, 6] = (1 / 4)
my_nullspace[13, 6] = (-1 / 2)

integrator2 = copy(integrate(lifting))
for j in 1:size(integrator2, 2)
    repteated = false
    for i in 1:size(integrator2, 1)
        if !iszero(integrator2[i, j]) && !repteated
            repteated = true
            @show i, j
        elseif !iszero(integrator2[i, j]) && repteated
            integrator2[i, j] = 0
            @show "zeroing", i, j
        end
    end
end

integrator_manual = hcat(integrator2, my_nullspace)

function find_initial_point()
    model = Model(with_optimizer(Hypatia.Optimizer{Float64}, verbose = true, use_dense = false))
    # model = Model(with_optimizer(Mosek.Optimizer, QUIET = false))
    @variable(model, coeffs[1:length(monos_sqr)])
    # @variable(model, t)
    @variable(model, hess[1:(length(monos_hess) * div(n * (n + 1), 2))])
    # hankel_old = lambda_direct(coeffs * 1)
    # hankel = monomial_lambda(lifting * inv(lifting' * lifting) * coeffs)
    N = nullspace(lifting')
    @constraints(model, begin
        lifting' * hess .== coeffs
        N' * hess .== zeros(size(N, 2))
        # my_nullspace' * hess .== zeros(6)
    end)
    hankel = monomial_lambda(hess * 1)
    s = size(hankel, 1)

    @constraint(model, [hankel[i, j] - (i == j ? 0.1 : 0) for i in 1:s for j in 1:i] in MOI.PositiveSemidefiniteConeTriangle(s))
    # @constraint(model, [t, 1,  [hankel[i, j] for i in 1:s for j in 1:i]...] in MOI.LogDetConeTriangle(s))

    # @SDconstraint(model, hankel ⪰ eps * I)
    # @objective(model, Max, t)
    optimize!(model)
    H = value.(hankel)
    H[abs.(H) .< 1e-9] .= 0
    H = Symmetric(H, :L)
    @show H
    @show dot(value.(coeffs), monos_sqr)
    @show isposdef(H)
    @show logdet(H)

end

hess = differentiate(dot(coeffs, monos_sqr), x, 2)
hankel_sym = Symmetric(hankel_old, :L)
dp = 0
for i in 1:n, j in 1:n, k in 1:length(monos_low), l in 1:length(monos_low)
    global dp += coefficient(hess[i, j], monos_low[k] * monos_low[l]) * hankel_sym[(i - 1) * 3 + k, (j - 1) * 3 + l] # * monos_low[k] * monos_low[l]
end


# 1.2000000000000002
# -6.523353691671332e-17
# 2.3698645623511286e-17
# 7.689634644949283
# 9.732396729288051e-16
# 7.689634644949283
# 1.5425503798568115e-16
# 1.2668625483510415e-15
# 1.5309340259959062e-15
# 8.00935267466167e-16
# 1.2000000000000002
# -2.2278572467804496e-16
# 7.689634644949298
# -5.319110456494835e-16
# 1.2000000000000002
# -0.0
# 4.490755470545132e-16
# -3.8339068559688606e-16
# 4.118413075418305e-17
# -6.70463770558626e-16
# -9.284211040019328e-16
# -0.0
# 1.1554912489174514e-15
# -6.709795971515719e-16
# -0.0
# 0.2
# -0.0
# -0.0
# 0.2
# -0.0
# 0.2
# arr = [1.2, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 10.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.2]


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





;
