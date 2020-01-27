#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

list of predefined signomials and domains from various applications


=#

signomials = Dict{Symbol, NamedTuple}(
    :denseunit1d => (n = 1,
        f = (z -> 1 + 2real(z[1]) + abs(z[1])^2 + 2real(z[1]^2) + 2real(z[1]^2 * conj(z[1])) + abs(z[1])^4),
        gs = [z -> 1 - abs2(z[1])],
        g_halfdegs = [1],
        truemin = 0,
        ),
    )
