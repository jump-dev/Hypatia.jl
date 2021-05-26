
nearestpolymat_n_d_ms = [
    [
    (1, 1, 2), # compile run
    (1, 20, 2),
    (1, 20, 4),
    (1, 20, 8),
    (1, 20, 16),
    (1, 20, 32),
    ],
    [
    (1, 1, 2), # compile run
    (1, 40, 2),
    (1, 40, 4),
    (1, 40, 8),
    (1, 40, 16),
    (1, 40, 32),
    ],
    [
    (1, 1, 2), # compile run
    (1, 80, 2),
    (1, 80, 4),
    (1, 80, 8),
    (1, 80, 16),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 2),
    (4, 2, 4),
    (4, 2, 8),
    (4, 2, 16),
    ],
    [
    (3, 1, 2), # compile run
    (4, 4, 2),
    (4, 4, 4),
    (4, 4, 8),
    ],
    ]

insts = OrderedDict()
insts["nat"] = (nothing, [[(n, d, m, false, true, false)
    for (n, d, m) in ndms] for ndms in nearestpolymat_n_d_ms])
insts["ext"] = (nothing, [[(n, d, m, true, false, false)
    for (n, d, m) in ndms] for ndms in nearestpolymat_n_d_ms])
return (NearestPolyMatJuMP, insts)
