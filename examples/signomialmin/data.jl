#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

list of predefined signomials and domains from various applications


=#

signomials = Dict{Symbol, NamedTuple}(
    :motzkin => ( # 1 - 3*x1^2*x2^2 + x1^2*x2^4 + x1^4*x2^2
        c = [1, -3, 1, 1],
        A = [0 0; 2 2; 2 4; 4 2],
        truemin = 0,
        ),
    )
