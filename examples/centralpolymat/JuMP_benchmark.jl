
centralpolymat_m_ks = [
    [
    (1, 4), # compile run
    (1, 40),
    (1, 80),
    (1, 120),
    (1, 160),
    ],
    [
    (2, 2), # compile run
    (2, 6),
    (2, 9),
    (2, 12),
    (2, 15),
    ],
    [
    (3, 2), # compile run
    (4, 3),
    (4, 4),
    (4, 5),
    ],
    [
    (3, 2), # compile run
    (8, 2),
    (8, 3),
    ],
    ]
centralpolymat_insts(ext::MatSpecExt) = [
    [(m, k, ext) for (m, k) in mks]
    for mks in centralpolymat_m_ks
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(
    centralpolymat_insts(MatNegEntropyConj()),
    centralpolymat_insts(MatNegSqrtConj()),
    centralpolymat_insts(MatPower12Conj(0.3)),
    centralpolymat_insts(MatPower12Conj(1.7)),
    ))
insts["ext"] = (nothing, vcat(
    centralpolymat_insts(MatNegEntropyConjEigOrd()),
    centralpolymat_insts(MatNegSqrtConjEigOrd()),
    centralpolymat_insts(MatPower12ConjEigOrd(0.3)),
    centralpolymat_insts(MatPower12ConjEigOrd(1.7)),
    ))
return (CentralPolyMatJuMP, insts)
