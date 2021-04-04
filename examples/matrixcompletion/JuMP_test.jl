
insts = Dict()
insts["minimal"] = [
    ((2, 2),),
    ((2, 2), ExpPSDOptimizer),
    ((2, 2), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    ((3, 4),),
    ((3, 4), ExpPSDOptimizer),
    ((3, 4), SOCExpPSDOptimizer),
    ((10, 3),),
    ((5, 3), ExpPSDOptimizer),
    ((5, 3), SOCExpPSDOptimizer),
    ]
insts["slow"] = [
    ((2, 12), SOCExpPSDOptimizer),
    ((10, 14),),
    ((10, 14), SOCExpPSDOptimizer),
    ((2, 40),),
    ((10, 18),),
    ]
insts["various"] = [
    ((10, 20),),
    ((10, 20), ExpPSDOptimizer),
    ((10, 20), SOCExpPSDOptimizer),
    ((10, 25),),
    ((10, 25), ExpPSDOptimizer),
    ((10, 25), SOCExpPSDOptimizer),
    ((10, 30),),
    ]
return (MatrixCompletionJuMP, insts)
