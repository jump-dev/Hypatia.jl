
matrixcompletion_insts = [
    [(k, d) for d in vcat(2, 5:5:max_d)] # includes compile run
    for (k, max_d) in ((10, 45), (20, 30))
    ]

insts = OrderedDict()
insts["nat"] = (nothing, matrixcompletion_insts)
insts["extEP"] = (:ExpPSD, matrixcompletion_insts)
insts["extSEP"] = (:SOCExpPSD, matrixcompletion_insts)
return (MatrixCompletionJuMP, insts)
