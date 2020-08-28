
insts = Dict()
insts["minimal"] = [
    ((2, 3),),
    ((2, 3), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((5, 8),),
    ((5, 8), StandardConeOptimizer),
    ((12, 20),),
    ]
insts["slow"] = [
    ((12, 24), StandardConeOptimizer),
    ((14, 140),),
    ((14, 140), StandardConeOptimizer),
    ((40, 70),),
    ((18, 180),),
    ]
return (MatrixCompletionJuMP, insts)
