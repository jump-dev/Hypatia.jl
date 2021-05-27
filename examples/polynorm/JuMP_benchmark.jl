
polynorm_l2_n_d_ms = [
    [
    (1, 1, 2), # compile run
    (1, 20, 4),
    (1, 20, 8),
    (1, 20, 16),
    (1, 20, 32),
    (1, 20, 64),
    ],
    [
    (1, 1, 2), # compile run
    (1, 40, 4),
    (1, 40, 8),
    (1, 40, 16),
    (1, 40, 32),
    (1, 40, 64),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 4),
    (4, 2, 8),
    (4, 2, 16),
    (4, 2, 32),
    ],
    [
    (3, 1, 2), # compile run
    (4, 4, 4),
    (4, 4, 8),
    (4, 4, 16),
    (4, 4, 32),
    ],
    ]

polynorm_l1_n_d_ms = [
    [
    (1, 1, 2), # compile run
    (1, 40, 8),
    (1, 40, 16),
    (1, 40, 32),
    (1, 40, 64),
    (1, 40, 128),
    ],
    [
    (1, 1, 2), # compile run
    (1, 80, 8),
    (1, 80, 16),
    (1, 80, 32),
    (1, 80, 64),
    (1, 80, 128),
    ],
    [
    (3, 1, 2), # compile run
    (4, 2, 8),
    (4, 2, 16),
    (4, 2, 32),
    (4, 2, 64),
    (4, 2, 128),
    ],
    [
    (3, 1, 2), # compile run
    (4, 4, 4),
    (4, 4, 8),
    (4, 4, 16),
    (4, 4, 32),
    (4, 4, 64),
    ],
    ]

polynorm_insts(
    use_l1::Bool,
    use_norm_cone::Bool,
    use_wsos::Bool,
    d_factors::Vector{Int},
    ) = [
    [(n, d, f * d, m, use_l1, use_norm_cone, use_wsos) for (n, d, m) in ndms]
    for f in d_factors
    for ndms in (use_l1 ? polynorm_l1_n_d_ms : polynorm_l2_n_d_ms)
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(
    polynorm_insts(false, true, false, [1, 2]),
    polynorm_insts(true, true, false, [1,]),
    ))
insts["ext"] = (nothing, vcat(
    polynorm_insts(false, false, true, [1, 2]),
    polynorm_insts(false, false, false, [1, 2]),
    polynorm_insts(true, false, true, [1,]),
    ))

return (PolyNormJuMP, insts)
