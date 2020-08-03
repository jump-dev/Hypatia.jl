
shapeconregr_n_ds = [
    (1, [
        3, # compile run
        10,
        20,
        30,
        40,
        50,
        60,
        ]),
    (2, [
        2, # compile run
        5,
        10,
        15,
        20,
        ]),
    (3, [
        1,
        2, # compile run
        4,
        6,
        8,
        ]),
    (4, [
        2,
        3,
        4,
        5,
        ]),
    (6, [
        2,
        3,
        ]),
    (8, [2,]),
    (10, [2,]),
    (12, [2,]),
    (14, [2,]),
    ]
shapeconregr_instances(use_nat::Bool) = [
    [(n, ceil(Int, 1.1 * binomial(n + 2d, n)), :func4, 100.0, 2d, use_nat, false, false, true, false) for d in ds]
    for (n, ds) in shapeconregr_n_ds
    ]

instances[ShapeConRegrJuMP]["nat"] = (nothing, shapeconregr_instances(true))
instances[ShapeConRegrJuMP]["ext"] = (StandardConeOptimizer, shapeconregr_instances(false))
