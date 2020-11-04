
rpm_n_d_m = [
    [
    (4, 1, 3), # compile run
    (4, 2, 3),
    (4, 4, 3),
    ],
    [
    (8, 1, 3), # compile run
    (8, 2, 3),
    ],
    [
    (4, 1, 5), # compile run
    (4, 2, 5),
    (4, 4, 5),
    ],
    [
    (8, 1, 5), # compile run
    (8, 2, 5),
    (8, 4, 5),
    ],
    [
    (2, 1, 10), # compile run
    (2, 2, 10),
    (2, 4, 10),
    ],
    [
    (4, 1, 10), # compile run
    (4, 2, 10),
    ],
    [
    (2, 1, 20), # compile run
    (2, 1, 20),
    (2, 2, 20),
    ],
    [
    (4, 1, 20), # compile run
    (4, 1, 20),
    ],
    [
    (2, 1, 30), # compile run
    (2, 1, 30),
    (2, 2, 30),
    ],
    [
    (4, 1, 30), # compile run
    (4, 1, 30),
    ],
    [
    (2, 1, 40), # compile run
    (2, 1, 40),
    (2, 2, 40),
    ],
    [
    (2, 1, 50), # compile run
    (2, 1, 50),
    ],
    [
    (2, 1, 60), # compile run
    (2, 1, 60),
    ],
    ]

insts = Dict()
insts["nat"] = (nothing, [[(n, d, m, false, true, false) for (n, d, m) in ndms] for ndms in rpm_n_d_m])
insts["ext"] = (nothing, [[(n, d, m, true, false, false) for (n, d, m) in ndms] for ndms in rpm_n_d_m])
return (RandomPolyMatJuMP, insts)
