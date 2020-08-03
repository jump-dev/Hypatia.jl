
matrixcompletion_instances = [
    [(d1, f * d1) for d1 in vcat(3, 10:5:50)] # includes compile run
    for f in (5, 10)
    ]

instances[MatrixCompletionJuMP]["nat"] = (nothing, matrixcompletion_instances)
instances[MatrixCompletionJuMP]["ext"] = (ExpPSDConeOptimizer, matrixcompletion_instances)
