#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

shapeconregr_n_ds = [
    [
        (2, 2), # compile run
        (2, 5),
        (2, 10),
        (2, 15),
        (2, 20),
        (2, 25),
        (2, 30),
    ],
    [
        (3, 2), # compile run
        (3, 4),
        (3, 6),
        (3, 8),
        (3, 10),
        (3, 12),
    ],
    [
        (3, 2), # compile run
        (4, 3),
        (4, 4),
        (4, 5),
        (4, 6),
        (4, 7),
    ],
    [
        (3, 2), # compile run
        (6, 2),
        (6, 3),
        (6, 4),
    ],
    [
        (3, 2), # compile run
        (8, 2),
        (8, 3),
    ],
    [
        (3, 2), # compile run
        (10, 2),
    ],
    [
        (3, 2), # compile run
        (12, 2),
    ],
    [
        (3, 2), # compile run
        (14, 2),
    ],
]
function shapeconregr_insts(use_nat::Bool)
    return [
        [
            (
                n,
                ceil(Int, 1.1 * binomial(n + 2d, n)),
                :func4,
                100.0,
                2d,
                use_nat,
                false,
                false,
                true,
                false,
            ) for (n, d) in nds
        ] for nds in shapeconregr_n_ds
    ]
end

insts = OrderedDict()
insts["nat"] = (nothing, shapeconregr_insts(true))
insts["ext"] = (:SOCExpPSD, shapeconregr_insts(false))
return (ShapeConRegrJuMP, insts)
