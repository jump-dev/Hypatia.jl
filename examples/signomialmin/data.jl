#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

list of predefined signomials and domains from various applications

prefixes:
- CS16 refers to "Relative Entropy Relaxations for Signomial Optimization" (2016) by Chandrasekaran & Shah
- MCW19 refers to "Signomial and Polynomial Optimization via Relative Entropy and Partial Dualization" (2019) by Murray, Chandrasekaran, & Wierman

see
https://rileyjmurray.github.io/sageopt/examples/examples.html
https://github.com/rileyjmurray/sageopt/blob/master/examples/MCW2019/ex4.py

c are coefficients
A are powers
x is a feasible point in domain, if known
obj_ub is an objective upper bound (value at feasible point), if known
=#

using Distributions

signomials = Dict{Symbol, NamedTuple}(
    :motzkin2 => ( # 1 - 3*x1^2*x2^2 + x1^2*x2^4 + x1^4*x2^2
        c = [1, -3, 1, 1],
        A = [0 0; 2 2; 2 4; 4 2],
        x = [],
        obj_ub = 0.0,
        ),
    :motzkin3 => ( # x3^6 - 3*x1^2*x2^2*x3^2 + x1^2*x2^4 + x1^4*x2^2
        c = [0, 1, -3, 1, 1],
        A = [0 0 0; 0 0 6; 2 2 2; 2 4 0; 4 2 0],
        x = [],
        obj_ub = 0.0,
        ),
    :CS16ex8_13 => ( # 10 exp{10.2x1} + 10 exp{9.8x2} + 10 exp{8.2x3} − 14.6794 exp{1.5089x1 + 1.0981x2 + 1.3419x3} − 7.8601 exp{1.0857x1 + 1.9069x2 + 1.6192x3} + 8.7838 exp{1.0459x1 + 0.0492x2 + 1.6245x3}
        c = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        A = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419; 1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245],
        x = [-0.3020, -0.2586, -0.4010],
        obj_ub = NaN,
        ),
    :CS16ex8_14 => ( # 10 exp{10.2x1} + 10 exp{9.8x2} + 10 exp{8.2x3} + 7.5907 exp{1.9864x1 + 0.2010x2 + 1.0855x3} − 10.9888 exp{2.8242x1 + 1.9355x2 + 2.0503x3} − 13.9164 exp{0.1828x1 + 2.7772x2 + 1.9001x3}
        c = [0, 10, 10, 10, 7.5907, -10.9888, -13.9164],
        A = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.9864 0.2010 1.0855; 2.8242 1.9355 2.0503; 0.1828 2.7772 1.9001],
        x = [],
        obj_ub = -0.739,
        ),
    :CS16ex18 => ( # 10 exp{10.2070x1 + 0.0082x2 − 0.0039x3} + 10 exp{−0.0081x1 + 9.8024x2 − 0.0097x3} + 10 exp{0.0070x1 − 0.0156x2 + 8.1923x3} − 14.6794 exp{1.5296x1 + 1.0927x2 + 1.3441x3} − 7.8601 exp{1.0750x1 + 1.9108x2 + 1.6339x3} + 8.7838 exp{1.0513x1 + 0.0571x2 + 1.6188x3}
        c = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        A = [0 0 0; 10.2070 0.0082 -0.0039; -0.0081 9.8024 -0.0097; 0.0070 -0.0156 8.1923; 1.5296 1.0927 1.3441; 1.0750 1.9108 1.6339; 1.0513 0.0571 1.6188],
        x = [-0.3020, -0.2586, -0.4010],
        obj_ub = NaN,
        ),
    )

eval_signomial(c::Vector, A::Matrix, x::Vector) = sum(c_k * exp(dot(A_k, x)) for (c_k, A_k) in zip(c, eachrow(A)))

# TODO use bounded domains
# TODO maybe use sparsity
function random_signomial(m::Int, n::Int; neg_c_frac::Real = 0.2, num_samples::Int = 20)
    c = rand(m)
    for k in 1:m
        if rand() < neg_c_frac
            c[k] *= -1
        end
    end
    A = randn(m, n)
    # sample points to get an objective upper bound
    obj_ub = eval_signomial(c, A, zeros(n))
    for i in 1:num_samples
        x = rand(Distributions.Normal(0, 10), n)
        obj_ub = min(obj_ub, eval_signomial(c, A, x))
    end
    return (c, A, obj_ub)
end
