
centralpolymat_m_ks = [
    [
    (1, 4), # compile run
    (1, 15),
    (1, 25),
    (1, 50),
    (1, 75),
    (1, 100),
    (1, 125),
    (1, 150),
    (1, 175),
    ],
    [
    (3, 2), # compile run
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),
    ],
    ]
centralpolymat_insts(ext::MatSpecExt) = [
    [(m, k, ext) for (m, k) in mks]
    for mks in centralpolymat_m_ks
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(centralpolymat_insts.([
    MatNegEntropy(),
    MatNegEntropyConj(),
    MatPower12(1.5),
    MatPower12Conj(1.5),
    ])...))
insts["ext"] = (nothing, vcat(centralpolymat_insts.([
    MatNegEntropyEigOrd(),
    MatNegEntropyConjEigOrd(),
    MatPower12EigOrd(1.5),
    MatPower12ConjEigOrd(1.5),
    ])...))
return (CentralPolyMatJuMP, insts)
