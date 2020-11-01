
l2_n_d_m = [
    (1, 1, 2), # compile run
    (4, 1, 3),
    (4, 2, 3),
    (5, 1, 3),
    (5, 2, 3),
    (6, 1, 3),
    (6, 2, 3),
    (4, 1, 5),
    (4, 2, 5),
    (5, 1, 5),
    (5, 2, 5),
    (4, 1, 10),
    (4, 2, 10),
    (5, 1, 10),
    (4, 1, 15),
    (4, 2, 15),
    (4, 1, 30),
    ]

l1_n_d_m = [
    (1, 1, 2), # compile run
    (4, 2, 3),
    (4, 4, 3),
    (5, 2, 3),
    (5, 4, 3),
    (6, 2, 3),
    (6, 4, 3),

    (4, 2, 5),
    (4, 4, 5),
    (5, 2, 5),
    (5, 4, 5),

    (4, 2, 10),
    (4, 4, 10),
    (5, 2, 10),
    (5, 4, 10),

    (4, 2, 15),
    (4, 4, 15),
    (4, 2, 30),
    (4, 4, 30),

    (2, 3, 20),
    (2, 4, 20),
    (4, 2, 20),
    (4, 3, 20),
    (4, 4, 20),

    (2, 3, 40),
    (2, 4, 40),
    (4, 2, 40),
    (4, 3, 40),

    (2, 2, 50),
    (2, 3, 50),
    (4, 2, 50),
    (4, 3, 50),

    (2, 2, 60),
    (2, 3, 60),
    (4, 2, 60),
    (4, 3, 60),

    (2, 2, 80),
    (2, 3, 80),
    (4, 2, 80),
    (4, 3, 80),
    ]

polynorml2_insts(use_l2::Bool) = [[
    (n, f * d, d, m, false, use_l2)
    for f in [1, 2] for (n, d, m) in l2_n_d_m
    ]]

polynorml1_insts(use_l1::Bool) = [[
    (n, 2 * d, d, m, true, use_l1)
    for (n, d, m) in l2_n_d_m
    ]]

insts = Dict()
insts["L2_WSOSL2"] = (nothing, polynorml2_insts(true))
insts["L2_WSOSPSD"] = (nothing, polynorml2_insts(false))
insts["L1_WSOSL1"] = (nothing, polynorml1_insts(true))
insts["L1_WSOS"] = (nothing, polynorml1_insts(false))
return (PolyNormJuMP, insts)
