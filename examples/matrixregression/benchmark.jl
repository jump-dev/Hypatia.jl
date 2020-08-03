
matrixregression_instances = [
    [(ceil(Int, 6m), m, 5m, 0, 0.2, 0, 0, 0) for m in vcat(3, 5:5:55)] # includes compile run
    ]

instances[MatrixRegressionJuMP]["nat"] = (nothing, matrixregression_instances)
instances[MatrixRegressionJuMP]["ext"] = (StandardConeOptimizer, matrixregression_instances)
