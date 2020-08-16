
matrixcompletion_insts = [
    [(d1, f * d1) for d1 in vcat(3, 10:5:50)] # includes compile run
    for f in (5, 10)
    ]

insts[MatrixCompletionJuMP]["nat"] = (nothing, matrixcompletion_insts)
insts[MatrixCompletionJuMP]["ext"] = (StandardConeOptimizer, matrixcompletion_insts)
