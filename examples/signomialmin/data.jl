#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

list of predefined signomials and domains from various applications

note: the domains are implicitly subsets of nonnegative orthant

prefix MCW19 refers to the paper "Signomial and Polynomial Optimization via Relative Entropy and Partial Dualization" (2019) by Murray, Chandrasekaran, & Wierman
=#

signomials = Dict{Symbol, NamedTuple}(
    :motzkin2 => ( # 1 - 3*x1^2*x2^2 + x1^2*x2^4 + x1^4*x2^2
        c = [1, -3, 1, 1],
        A = [0 0; 2 2; 2 4; 4 2],
        truemin = 0,
        ),
    :motzkin3 => ( # x3^6 - 3*x1^2*x2^2*x3^2 + x1^2*x2^4 + x1^4*x2^2
        c = [0, 1, -3, 1, 1],
        A = [0 0 0; 0 0 6; 2 2 2; 2 4 0; 4 2 0],
        truemin = 0,
        ),
    :MCW19ex8_13 => ( # 10 exp{10.2x1} + 10 exp{9.8x2} + 10 exp{8.2x3} − 14.6794 exp{1.5089x1 + 1.0981x2 + 1.3419x3} − 7.8601 exp{1.0857x1 + 1.9069x2 + 1.6192x3} + 8.7838 exp{1.0459x1 + 0.0492x2 + 1.6245x3}
        c = [0, 10, 10, 10, -14.6794, -7.8601, 8.7838],
        A = [0 0 0; 10.2 0 0; 0 9.8 0; 0 0 8.2; 1.5089 1.0981 1.3419; 1.0857 1.9069 1.6192; 1.0459 0.0492 1.6245],
        truemin = -0.9747,
        ),
    )
