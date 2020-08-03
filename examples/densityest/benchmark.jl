
densityest_n_ds = [
    (1, [
        4, # compile run
        75,
        150,
        300,
        600,
        900,
        1200,
        1500,
        ]),
    (2, [
        3, # compile run
        10,
        20,
        30,
        40,
        ]),
    (3, [
        2, # compile run
        3,
        6,
        9,
        12,
        ]),
    (4, [
        2,
        4,
        6,
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
densityest_instances(use_nat::Bool) = [
    [(ceil(Int, 1.1 * binomial(n + 2d, n)), n, 2d, use_nat) for d in ds]
    for (n, ds) in densityest_n_ds
    ]

instances[DensityEstJuMP]["nat"] = (nothing, densityest_instances(true))
instances[DensityEstJuMP]["ext"] = (ExpPSDConeOptimizer, densityest_instances(false))
