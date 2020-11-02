
polynorm_l2_n_d_ms = [
    [
    (3, 1, 2), # compile run
    (4, 2, 3),
    (4, 4, 3),
    ],
    [
    (3, 1, 2), # compile run
    (5, 2, 3),
    (5, 4, 3),
    ],
    [
    (3, 1, 2), # compile run
    (6, 2, 3),
    (6, 4, 3),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 5),
    (4, 4, 5),
    ],
    [
    (3, 1, 2), # compile run
    (5, 2, 5),
    (5, 4, 5),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 10),
    (4, 4, 10),
    ],
    [
    (3, 1, 2), # compile run
    (5, 2, 10),
    (5, 4, 10),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 15),
    (4, 4, 15),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 30),
    (4, 4, 30),
    ],
    ]

polynorm_l1_n_d_ms = [
    [
    (3, 1, 2), # compile run
    (4, 2, 3),
    (4, 4, 3),
    ],
    [
    (3, 1, 2), # compile run
    (5, 2, 3),
    (5, 4, 3),
    ],
    [
    (3, 1, 2), # compile run
    (6, 2, 3),
    (6, 4, 3),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 5),
    (4, 4, 5),
    ],
    [
    (3, 1, 2), # compile run
    (5, 2, 5),
    (5, 4, 5),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 10),
    (4, 4, 10),
    ],
    [
    (3, 1, 2), # compile run
    (5, 2, 10),
    (5, 4, 10),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 15),
    (4, 4, 15),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 30),
    (4, 4, 30),
    ],
    [
    (2, 1, 2), # compile run
    (2, 3, 20),
    (2, 4, 20),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 20),
    (4, 3, 20),
    (4, 4, 20),
    ],
    [
    (2, 1, 2), # compile run
    (2, 3, 40),
    (2, 4, 40),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 40),
    (4, 3, 40),
    ],
    [
    (2, 1, 2), # compile run
    (2, 2, 50),
    (2, 3, 50),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 50),
    (4, 3, 50),
    ],
    [
    (2, 1, 2), # compile run
    (2, 2, 60),
    (2, 3, 60),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 60),
    (4, 3, 60),
    ],
    [
    (2, 1, 2), # compile run
    (2, 2, 80),
    (2, 3, 80),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 80),
    (4, 3, 80),
    ],
    ]

polynorm_insts(use_l1::Bool, use_norm_cone::Bool, d_factors::Vector{Int}) = [
    [(n, f * d, d, m, use_l1, use_norm_cone) for (n, d, m) in ndms]
    for f in d_factors
    for ndms in (use_l1 ? polynorm_l1_n_d_ms : polynorm_l2_n_d_ms)
    ]

insts = Dict()
insts["nat"] = (nothing, vcat(
    polynorm_insts(false, true, [1, 2]),
    polynorm_insts(false, false, [1, 2]),
    polynorm_insts(true, true, [1,]),
    polynorm_insts(true, false, [1,]),
    ))
return (PolyNormJuMP, insts)
