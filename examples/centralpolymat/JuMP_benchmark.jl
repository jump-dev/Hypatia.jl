
centralpolymat_m_ks = [
    [
    (1, 4), # compile run
    (1, 25),
    (1, 50),
    (1, 75),
    (1, 100),
    (1, 125),
    (1, 150),
    (1, 175),
    (1, 200),
    ],
    [
    (3, 2), # compile run
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
    ],
    ]
centralpolymat_insts(ext::MatSpecExt) = [
    [(m, k, ext) for (m, k) in mks]
    for mks in centralpolymat_m_ks
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(
    centralpolymat_insts(MatNegGeom()),
    centralpolymat_insts(MatNegEntropyConj()),
    centralpolymat_insts(MatPower12(1.5)),
    centralpolymat_insts(MatPower12Conj(1.5)),
    ))
insts["ext"] = (nothing, vcat(
    centralpolymat_insts(MatNegGeomEFExp()),
    centralpolymat_insts(MatNegEntropyConjEigOrd()),
    centralpolymat_insts(MatPower12EigOrd(1.5)),
    centralpolymat_insts(MatPower12ConjEigOrd(1.5)),
    ))
return (CentralPolyMatJuMP, insts)
