
insts = Dict()
insts["minimal"] = [
    ((1, 2, false),),
    ((1, 2, false), SOCExpPSDOptimizer),
    ((2, 2, true),),
    ((2, 2, true), ExpPSDOptimizer),
    ]
insts["fast"] = [
    ((1, 10, true),),
    ((1, 10, true), SOCExpPSDOptimizer),
    ((1, 15, false),),
    ((1, 15, false), ExpPSDOptimizer),
    ((2, 3, true),),
    ((2, 3, true), SOCExpPSDOptimizer),
    ((2, 3, false),),
    ((2, 3, false), ExpPSDOptimizer),
    ((2, 6, true),),
    ((2, 5, true), SOCExpPSDOptimizer),
    ((2, 7, false),),
    ((2, 6, false), SOCExpPSDOptimizer),
    ((3, 2, true),),
    ((3, 2, false),),
    ((3, 4, true),),
    ((3, 4, false),),
    ((7, 2, true),),
    ((7, 2, true), ExpPSDOptimizer),
    ((7, 2, false),),
    ((7, 2, false), SOCExpPSDOptimizer),
    ]
insts["slow"] = [
    ((1, 20, false),),
    ((2, 3, false),),
    ((2, 10, false),),
    ((2, 8, false), SOCExpPSDOptimizer),
    ((3, 4, true), ExpPSDOptimizer),
    ((3, 4, false), ExpPSDOptimizer),
    ((3, 5, true),),
    ((3, 5, false),),
    ((6, 3, true),),
    ((6, 3, false),),
    ]
insts["various"] = [
    ((2, 5, true),),
    ((2, 5, true), ExpPSDOptimizer),
    ((2, 5, false),),
    ((2, 5, false), SOCExpPSDOptimizer),
    ((2, 5, false), ExpPSDOptimizer),
    ((2, 10, true),),
    ((2, 10, true), ExpPSDOptimizer),
    ((2, 10, false),),
    ((2, 10, false), SOCExpPSDOptimizer),
    ((2, 10, false), ExpPSDOptimizer),
    ((6, 3, true),),
    ((6, 3, false),),
    ]
return (CentralPolyMatJuMP, insts)
