#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

list of predefined signomials and domains from various applications

prefixes:
- CS16 refers to "Relative Entropy Relaxations for Signomial Optimization" (2016) by Chandrasekaran & Shah
- MCW19 refers to "Signomial and Polynomial Optimization via Relative Entropy and Partial Dualization" (2019) by Murray, Chandrasekaran, & Wierman

c are coefficients
A are powers
x is a feasible point in domain, if known
obj_ub is an objective upper bound (value at feasible point), if known
=#

using Distributions
using SparseArrays

signomialmin_data = Dict{Symbol, NamedTuple}(
    :motzkin2 => (
        # f = 1 - 3*x1^2*x2^2 + x1^2*x2^4 + x1^4*x2^2
        fc = [1, -3, 1, 1],
        fA = [0 0; 2 2; 2 4; 4 2],
        gc = [],
        gA = [],
        x = [],
        obj_ub = 0.0,
        ),
    :motzkin3 => (
        # f = x3^6 - 3*x1^2*x2^2*x3^2 + x1^2*x2^4 + x1^4*x2^2
        fc = [0, 1, -3, 1, 1],
        fA = [0 0 0; 0 0 6; 2 2 2; 2 4 0; 4 2 0],
        gc = [],
        gA = [],
        x = [],
        obj_ub = 0.0,
        ),
    :CS16ex8_13 => (
        # f = 10 exp(10.2x1) + 10 exp(9.8x2) + 10 exp(8.2x3) - 14.6794 exp(1.5089x1 + 1.0981x2 + 1.3419x3) - 7.8601 exp(1.0857x1 + 1.9069x2 + 1.6192x3) + 8.7838 exp(1.0459x1 + 0.0492x2 + 1.6245x3)
        fc = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        fA = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419; 1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245],
        gc = [],
        gA = [],
        x = [-0.3020, -0.2586, -0.4010],
        obj_ub = NaN,
        ),
    :CS16ex8_14 => (
        # f = 10 exp(10.2x1) + 10 exp(9.8x2) + 10 exp(8.2x3) + 7.5907 exp(1.9864x1 + 0.2010x2 + 1.0855x3) - 10.9888 exp(2.8242x1 + 1.9355x2 + 2.0503x3) - 13.9164 exp(0.1828x1 + 2.7772x2 + 1.9001x3)
        fc = [0, 10, 10, 10, 7.5907, -10.9888, -13.9164],
        fA = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.9864 0.2010 1.0855; 2.8242 1.9355 2.0503; 0.1828 2.7772 1.9001],
        gc = [],
        gA = [],
        x = [],
        obj_ub = -0.739,
        ),
    :CS16ex18 => (
        # f = 10 exp(10.2070x1 + 0.0082x2 - 0.0039x3) + 10 exp(-0.0081x1 + 9.8024x2 - 0.0097x3) + 10 exp(0.0070x1 - 0.0156x2 + 8.1923x3) - 14.6794 exp(1.5296x1 + 1.0927x2 + 1.3441x3) - 7.8601 exp(1.0750x1 + 1.9108x2 + 1.6339x3) + 8.7838 exp(1.0513x1 + 0.0571x2 + 1.6188x3)
        fc = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        fA = [0 0 0; 10.2070 0.0082 -0.0039; -0.0081 9.8024 -0.0097; 0.0070 -0.0156 8.1923; 1.5296 1.0927 1.3441; 1.0750 1.9108 1.6339; 1.0513 0.0571 1.6188],
        gc = [],
        gA = [],
        x = [-0.3020, -0.2586, -0.4010],
        obj_ub = NaN,
        ),
    :CS16ex12 => (
        # f = 10 exp(10.2x1) + 10 exp(9.8x2) + 10 exp(8.2x3) - 14.6794 exp(1.5089x1 + 1.0981x2 + 1.3419x3) - 7.8601 exp(1.0857x1 + 1.9069x2 + 1.6192x3) + 8.7838 exp(1.0459x1 + 0.0492x2 + 1.6245x3)
        # g1 = 1 - (8 exp(10.2x1) + 8 exp(9.8x2) + 8 exp(8.2x3) + 6.4 exp(1.0857x1 + 1.9069x2 + 1.6192x3)) >= 0
        fc = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        fA = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419; 1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245],
        gc = [[1, -8, -8, -8, -6.4]],
        gA = [[0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.0857 1.9069 1.6192]],
        x = [-0.4313, -0.3824, -0.6505],
        obj_ub = NaN,
        ),
    :CS16ex13 => (
        # f = 10 exp(10.2x1) + 10 exp(9.8x2) + 10 exp(8.2x3) - 14.6794 exp(1.5089x1 + 1.0981x2 + 1.3419x3) - 7.8601 exp(1.0857x1 + 1.9069x2 + 1.6192x3) + 8.7838 exp(1.0459x1 + 0.0492x2 + 1.6245x3)
        # g1 = -8 exp(10.2x1) - 8 exp(9.8x2) - 8 exp(8.2x3) + 0.7410 exp(1.5089x1 + 1.0981x2 + 1.3419x3) - 0.4492 exp(1.0857x1 + 1.9069x2 + 1.6192x3) + 1.4240 exp(1.0459x1 + 0.0492x2 + 1.6245x3) >= 0
        fc = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        fA = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419; 1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245],
        gc = [[-8, -8, -8, 0.7410, -0.4492, 1.4240]],
        gA = [[10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419; 1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245]],
        x = [],
        obj_ub = -0.7372,
        ),
    :MCW19ex1_mod => ( # removed -5 exp(-x2) term in f due to unbounded Lagrangian issue discussed in paper
        # f = 0.5 exp(x1 - x2) - exp(x1)
        # g1 = 100 - exp(x2 - x3) - exp(x2) - 0.05 exp(x1 + x3) >= 0
        # g2:4 = -(70, 1, 0.5) .+ exp.(x) .>= 0
        # g5:7 = (150, 30, 21) .- exp.(x) .>= 0
        fc = [0, 0.5, -1, 0],
        fA = [0 0 0; 1 -1 0; 1 0 0; 0 -1 0],
        gc = [[100, -1, -1, -0.05], [-70, 1], [-1, 1], [-0.5, 1], [150, -1], [30, -1], [21, -1]],
        gA = [[0 0 0; 0 1 -1; 0 1 0; 1 0 1], [0 0 0; 1 0 0], [0 0 0; 0 1 0], [0 0 0; 0 0 1], [0 0 0; 1 0 0], [0 0 0; 0 1 0], [0 0 0; 0 0 1]],
        x = [5.01063529, 3.40119660, -0.48450710],
        obj_ub = NaN,
        ),
    :MCW19ex8 => (
        # f = 0.05 exp(x1) + 0.05 exp(x2) + 0.05 exp(x3) + exp(x9)
        # g1 = 1 + 0.5 exp(x1 + x4 - x7) - exp(x10 - x7) >= 0
        # g2 = 1 + 0.5 exp(x2 + x5 - x8) - exp(x7 - x8) >= 0
        # g3 = 1 + 0.5 exp(x3 + x6 - x9) - exp(x8 - x9) >= 0
        # g4 = 1 - 0.25 exp(-x10) - 0.5 exp(x9 - x10) >= 0
        # g5 = 1 - 0.79681 exp(x4 - x7) >= 0
        # g6 = 1 - 0.79681 exp(x5 - x8) >= 0
        # g7 = 1 - 0.79681 exp(x6 - x9) >= 0
        fc = [0, 0.05, 0.05, 0.05, 1],
        fA = sparse([2, 3, 4, 5], [1, 2, 3, 9], [1, 1, 1, 1], 5, 10),
        gc = [[1, 0.5, -1], [1, 0.5, -1], [1, 0.5, -1], [1, -0.25, -0.5], [1, -0.79681], [1, -0.79681], [1, -0.79681]],
        gA = [
            sparse([2, 2, 2, 3, 3], [1, 4, 7, 10, 7], [1, 1, -1, 1, -1], 3, 10),
            sparse([2, 2, 2, 3, 3], [2, 5, 8, 7, 8], [1, 1, -1, 1, -1], 3, 10),
            sparse([2, 2, 2, 3, 3], [3, 6, 9, 8, 9], [1, 1, -1, 1, -1], 3, 10),
            sparse([2, 3, 3], [10, 9, 10], [-1, 1, -1], 3, 10),
            sparse([2, 2], [4, 7], [1, -1], 2, 10),
            sparse([2, 2], [5, 8], [1, -1], 2, 10),
            sparse([2, 2], [6, 9], [1, -1], 2, 10),
            ],
        x = [],
        obj_ub = 0.2056534,
        ),
    )

eval_signomial(c::Vector, A::AbstractMatrix, x::Vector) = sum(c_k * exp(dot(A_k, x)) for (c_k, A_k) in zip(c, eachrow(A)))

function random_signomial(
    m::Int,
    n::Int;
    neg_c_frac::Real = 0.0,
    sparsity::Real = 0.3,
    )
    c = rand(m)
    for k in 1:m
        if rand() < neg_c_frac
            c[k] *= -1
        end
    end
    A = vcat(zeros(1, n), Matrix(sprandn(m - 1, n, sparsity)))
    for k in 2:size(A, 1)
        if iszero(norm(A[k, :]))
            A[k, :] = randn(n)
        end
    end
    return (c, A)
end

function random_instance(
    m::Int,
    n::Int;
    num_samples::Int = 100,
    neg_c_frac::Real = 0.0,
    sparsity::Real = 0.3,
    )
    # random objective signomial
    (fc, fA) = random_signomial(m, n, neg_c_frac = neg_c_frac, sparsity = sparsity)

    # bounded domain set (in exp space) is intersection of positive orthant and ball
    gc = [vcat(1, fill(-1, n))]
    gA = [sparse(2:(n + 1), 1:n, fill(2, n))]

    # sample points to get an objective upper bound
    obj_ub = Inf
    for i in 1:num_samples
        x = abs.(randn(n))
        r = rand(Distributions.Exponential(0.5))
        x /= sqrt(sum(abs2, x) + r)
        x = log.(x)
        @assert eval_signomial(gc[1], gA[1], x) >= 0
        obj_ub = min(obj_ub, eval_signomial(fc, fA, x))
    end

    return (fc, fA, gc, gA, obj_ub)
end
