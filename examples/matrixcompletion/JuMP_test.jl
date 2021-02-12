
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
    ((3, 4),),
    ((3, 4), ExpPSDOptimizer),
    ((3, 4), SOCExpPSDOptimizer),
    ((6, 8),),
    ((6, 8), ExpPSDOptimizer),
    ((6, 8), SOCExpPSDOptimizer),
    ((12, 16),),
    ((12, 16), ExpPSDOptimizer),
    ((12, 16), SOCExpPSDOptimizer),
    ]
return (MatrixCompletionJuMP, insts)
