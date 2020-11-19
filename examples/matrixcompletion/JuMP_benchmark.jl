
matrixcompletion_insts = [
    [(k, d) for d in vcat(3, 10:5:max_d)] # includes compile run
    for (k, max_d) in ((5, 60), (10, 45))
    ]

insts = Dict()
insts["nat"] = (nothing, matrixcompletion_insts)
insts["ext"] = (SOCExpPSDOptimizer, matrixcompletion_insts)
return (MatrixCompletionJuMP, insts)
