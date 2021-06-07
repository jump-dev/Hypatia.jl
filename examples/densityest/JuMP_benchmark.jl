
densityest_n_ds = [
    [
    (1, 3), # compile run
    (1, 125),
    (1, 250),
    (1, 500),
    (1, 1000),
    (1, 2000),
    (1, 3000),
    ],
    [
    (2, 2), # compile run
    (2, 10),
    (2, 20),
    (2, 30),
    (2, 40),
    (2, 50),
    ],
    [
    (3, 2), # compile run
    (3, 6),
    (3, 9),
    (3, 12),
    (3, 15),
    ],
    [
    (3, 2), # compile run
    (4, 4),
    (4, 6),
    (4, 8),
    ],
    [
    (3, 2), # compile run
    (8, 2),
    (8, 3),
    ],
    ]
densityest_insts(use_nat::Bool) = [
    [(500, n, 2d, false, use_nat, use_nat) for (n, d) in nds]
    for nds in densityest_n_ds
    ]

insts = OrderedDict()
insts["nat"] = (nothing, densityest_insts(true))
insts["ext"] = (:SOCExpPSD, densityest_insts(false))
return (DensityEstJuMP, insts)
