
polymin_n_ds = [
    (1, [
        4, # compile run
        100,
        200,
        500,
        1000,
        1500,
        2500,
        3500,
        ]),
    (2, [
        3, # compile run
        15,
        30,
        45,
        60,
        ]),
    (3, [
        2, # compile run
        3,
        6,
        9,
        12,
        15,
        ]),
    (4, [
        4,
        6,
        8,
        ]),
    (8, [
        2,
        3,
        ]),
    (16, [
        1,
        2,
        ]),
    (32, [1,]),
    (64, [1,]),
    ]
polymin_insts(use_nat::Bool) = [
    [(n, d, false, use_nat) for d in ds]
    for (n, ds) in polymin_n_ds
    ]

insts[PolyMinJuMP]["nat"] = (nothing, polymin_insts(true))
insts[PolyMinJuMP]["ext"] = (StandardConeOptimizer, polymin_insts(false))
