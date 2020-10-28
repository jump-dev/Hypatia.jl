
insts = Dict()
insts["minimal"] = [
    ((2, 2),),
    ((2, 2), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((3, 4),),
    ((3, 4), StandardConeOptimizer),
    ((4, 5),),
    ]
insts["slow"] = [
    ((2, 12), StandardConeOptimizer),
    ((10, 14),),
    ((10, 14), StandardConeOptimizer),
    ((2, 40),),
    ((10, 18),),
    ]
return (MatrixCompletionJuMP, insts)
