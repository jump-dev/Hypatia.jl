
polymin_n_ds = [
    [
    (1, 4), # compile run
    (1, 100),
    (1, 200),
    (1, 500),
    (1, 1000),
    (1, 1500),
    (1, 2500),
    (1, 3500),
    ],
    [
    (2, 3), # compile run
    (2, 15),
    (2, 30),
    (2, 45),
    (2, 60),
    ],
    [
    (3, 2), # compile run
    (3, 3),
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
    [
    (3, 2), # compile run
    (16, 1),
    (16, 2),
    ],
    [
    (3, 2), # compile run
    (32, 1),
    ],
    [
    (3, 2), # compile run
    (64, 1),
    ],
    ]
polymin_insts(use_nat::Bool) = [
    [(n, d, false, use_nat) for (n, d) in nds]
    for nds in polymin_n_ds
    ]

insts = Dict()
insts["nat"] = (nothing, polymin_insts(true))
insts["ext"] = (StandardConeOptimizer, polymin_insts(false))
return (PolyMinJuMP, insts)
