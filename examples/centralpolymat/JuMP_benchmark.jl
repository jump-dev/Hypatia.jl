
centralpolymat_m_halfdegs = [
    [
    (1, 4), # compile run
    (1, 50),
    (1, 100),
    (1, 150),
    (1, 200),
    (1, 250),
    ],
    [
    (2, 2), # compile run
    (2, 3),
    (2, 6),
    (2, 9),
    (2, 12),
    (2, 15),
    ],
    [
    (3, 2), # compile run
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),
    ],
    [
    (3, 2), # compile run
    (8, 2),
    (8, 3),
    ],
    ]
centralpolymat_insts(ext::MatSpecExt) = [
    [(m, halfdeg, ext) for (m, halfdeg) in mhalfdegs]
    for mhalfdegs in centralpolymat_m_halfdegs
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(
    centralpolymat_insts(MatNegExp1()),
    centralpolymat_insts(MatPower12Conj(1.5)),
    ))
insts["ext"] = (nothing, vcat(
    centralpolymat_insts(MatNegExp1EigOrd()),
    centralpolymat_insts(MatPower12ConjEigOrd(1.5)),
    ))
return (CentralPolyMatJuMP, insts)
