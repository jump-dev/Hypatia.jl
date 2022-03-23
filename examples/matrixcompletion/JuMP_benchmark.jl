
matrixcompletion_insts = [
    # [(false, false, false, d, k * d, 0.8) for d in vcat(2, 5:5:max_d)] # includes compile run
    [(false, false, false, d, k * d, 0.8) for d in vcat(2, max_d)] # includes compile run
    for (k, max_d) in ((10, 70), (20, 55))
    ]

insts = OrderedDict()
insts["nat"] = (nothing, matrixcompletion_insts)
insts["extEP"] = (:ExpPSD, matrixcompletion_insts)
insts["extSEP"] = (:SOCExpPSD, matrixcompletion_insts)
return (MatrixCompletionJuMP, insts)
