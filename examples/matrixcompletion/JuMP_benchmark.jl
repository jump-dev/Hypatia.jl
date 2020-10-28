
matrixcompletion_insts = [
    [(k, d) for d in vcat(3, 10:5:50)] # includes compile run
    for k in (5, 10)
    ]

insts = Dict()
insts["nat"] = (nothing, matrixcompletion_insts)
insts["ext"] = (StandardConeOptimizer, matrixcompletion_insts)
return (MatrixCompletionJuMP, insts)
