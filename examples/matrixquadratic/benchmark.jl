
matrixquadratic_insts(use_nat::Bool) = [
    [(d1, 5d1, use_nat) for d1 in vcat(3, 5:5:60)] # includes compile run
    ]

insts = Dict()
insts["nat"] = (nothing, matrixquadratic_insts(true))
insts["ext"] = (StandardConeOptimizer, matrixquadratic_insts(false))
return (MatrixQuadraticJuMP, insts)
