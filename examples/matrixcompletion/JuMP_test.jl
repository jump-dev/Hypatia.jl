
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
    ((2, 4),),
    ((2, 4), ExpPSDOptimizer),
    ((2, 4), SOCExpPSDOptimizer),
    ((4, 8),),
    ((4, 8), ExpPSDOptimizer),
    ((4, 8), SOCExpPSDOptimizer),
    ((8, 16),),
    ((8, 16), ExpPSDOptimizer),
    ((8, 16), SOCExpPSDOptimizer),
    ]
return (MatrixCompletionJuMP, insts)
