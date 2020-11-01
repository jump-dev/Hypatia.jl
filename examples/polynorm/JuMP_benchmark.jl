
l2_n_d_m = [
    [
    (1, 1, 2), # compile run
    (4, 2, 3),
    (4, 4, 3),
    ],
    [
    (1, 1, 2), # compile run
    (5, 2, 3),
    (5, 4, 3),
    ],
    [
    (1, 1, 2), # compile run
    (6, 2, 3),
    (6, 4, 3),
    ],
    [
    (1, 1, 2), # compile run
    (4, 2, 5),
    (4, 4, 5),
    ],
    [
    (5, 2, 5),
    (5, 4, 5),
    ],
    [
    (1, 1, 2), # compile run
    (4, 2, 10),
    (4, 4, 10),
    ],
    [
    (1, 1, 2), # compile run
    (5, 2, 10),
    ],
    [
    (1, 1, 2), # compile run
    (4, 2, 15),
    (4, 4, 15),
    ],
    [
    (1, 1, 2), # compile run
    (4, 2, 30),
    ],
    ]

l1_n_d_m = [
    [
    (1, 1, 2), # compile run
    (4, 2, 3),
    (4, 4, 3),
    ],
    [
    (1, 1, 2), # compile run
    (5, 2, 3),
    (5, 4, 3),
    ],
    [
    (1, 1, 2), # compile run
    (6, 2, 3),
    (6, 4, 3),
    ],
    [
    (1, 1, 2), # compile run
    (4, 2, 5),
    (4, 4, 5),
    ],
    [
    (5, 2, 5),
    (5, 4, 5),
    ],
    [
    (1, 1, 2), # compile run
    (4, 2, 10),
    (4, 4, 10),
    ],
    [
    (1, 1, 2), # compile run
    (5, 2, 10),
    (5, 4, 10), # not in l2
    ],
    [
    (1, 1, 2), # compile run
    (4, 2, 15),
    (4, 4, 15),
    ],
    [
    (1, 1, 2), # compile run
    (4, 2, 30),
    (4, 4, 30), # not in l2
    ],
    [
    (1, 1, 2), # compile run
    (2, 3, 20),
    (2, 4, 20),
    (4, 2, 20),
    (4, 3, 20),
    (4, 4, 20),
    ],
    [
    (1, 1, 2), # compile run
    (2, 3, 40),
    (2, 4, 40),
    (4, 2, 40),
    (4, 3, 40),
    ],
    [
    (1, 1, 2), # compile run
    (2, 2, 50),
    (2, 3, 50),
    (4, 2, 50),
    (4, 3, 50),
    ],
    [
    (1, 1, 2), # compile run
    (2, 2, 60),
    (2, 3, 60),
    (4, 2, 60),
    (4, 3, 60),
    ],
    [
    (1, 1, 2), # compile run
    (2, 2, 80),
    (2, 3, 80),
    (4, 2, 80),
    (4, 3, 80),
    ],
    ]

polynorml2_insts(use_l2::Bool) = [
    [(n, f * d, d, m, false, use_l2) for (n, d, m) in ndms]
    for f in [1, 2] for ndms in l2_n_d_m
    ]

polynorml1_insts(use_l1::Bool) = [
    [(n, d, d, m, true, use_l1) for (n, d, m) in ndms]
    for ndms in l1_n_d_m
    ]

insts = Dict()
insts["nat"] = (nothing, vcat(
    polynorml1_insts(true),
    polynorml1_insts(false),
    polynorml2_insts(true),
    polynorml2_insts(false),
    ))
return (PolyNormJuMP, insts)
