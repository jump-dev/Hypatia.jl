
matrixquadratic_instances(use_nat::Bool) = [
    [(d1, 5d1, use_nat) for d1 in vcat(3, 5:5:60)] # includes compile run
    ]

instances[MatrixQuadraticJuMP]["nat"] = (nothing, matrixquadratic_instances(true))
instances[MatrixQuadraticJuMP]["ext"] = (StandardConeOptimizer, matrixquadratic_instances(false))
