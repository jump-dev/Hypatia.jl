
# size is symmetric wrt n and d
centralpolymat_nds = [
    (1, 2),
    (1, 50), (1, 100), (1, 200),
    (2, 4), (2, 8), (2, 16),
    (3, 2), (3, 4), (3, 8),
    (4, 4), (4, 5),
    ]
centralpolymat_insts(ext::MatSpecExt) = [
    [(n, d, ext) for (n, d) in centralpolymat_nds] # includes compile run
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
