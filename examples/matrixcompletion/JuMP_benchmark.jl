
matrixcompletion_insts = [
    [(d1, f * d1) for d1 in vcat(3, 10:5:50)] # includes compile run
    for f in (5, 10)
    ]

insts = Dict()
insts["nat"] = (nothing, matrixcompletion_insts)
insts["ext"] = (StandardConeOptimizer, matrixcompletion_insts)
return (MatrixCompletionJuMP, insts)
