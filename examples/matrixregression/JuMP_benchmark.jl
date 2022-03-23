
matrixregression_insts = [
    vcat((4, 3), [(n, m) for n in ns]) for (m, ns) in ( # includes compile run
        (15, [30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000]),
        (30, [30, 100, 300, 1000, 3000, 10000, 30000, 100000]),
        )
    ]

insts = OrderedDict()
insts["nat"] = (nothing, matrixregression_insts)
insts["ext"] = (:SOCExpPSD, matrixregression_insts)
return (MatrixRegressionJuMP, insts)
