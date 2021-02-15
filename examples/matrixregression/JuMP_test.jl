
insts = Dict()
insts["minimal"] = [
    ((2, 3, 4, false, 0, 0, 0, 0, 0),),
    ((5, 3, 4, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((5, 3, 4, false, 0.1, 0.1, 0.1, 0.2, 0.2), SOCExpPSDOptimizer),
    ((5, 4, 4, true, 0.1, 0, 0, 0, 0),),
    ((5, 3),),
    ((5, 3), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    ((3, 4, 5, false, 0, 0, 0, 0, 0),),
    ((3, 4, 5, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((3, 4, 5, false, 0, 0.1, 0.1, 0, 0),),
    ((20, 5),),
    ((6, 5), SOCExpPSDOptimizer),
    ((10, 20, 20, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((20, 10, 20, true, 0.1, 0, 0, 0, 0),),
    ((50, 8, 12, false, 0, 0, 0, 0, 0),),
    ((50, 8, 12, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((50, 8, 12, false, 0.1, 0.1, 0.1, 0.2, 0.2), SOCExpPSDOptimizer),
    ((15, 8, 12, true, 0.1, 0, 0, 0, 0),),
    ((15, 8, 8, true, 0, 0, 0.1, 0, 0), SOCExpPSDOptimizer),
    ]
insts["slow"] = [
    ((15, 20, 50, false, 0, 0, 0, 0, 0),),
    ((15, 20, 50, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((15, 20, 50, false, 0.1, 0.1, 0.1, 0.2, 0.2), SOCExpPSDOptimizer),
    ((15, 20, 50, false, 0, 0.1, 0.1, 0, 0),),
    ]
insts["various"] = [
    ((3, 2, 5, true, 0.1, 0.1, 0.1, 0.2, 0.2), SOCExpPSDOptimizer),
    ((3, 2, 5, true, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((3, 2, 5, false, 0.1, 0.1, 0.1, 0.2, 0.2), SOCExpPSDOptimizer),
    ((3, 2, 5, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((6, 4, 10, true, 0.1, 0.1, 0.1, 0.2, 0.2), SOCExpPSDOptimizer),
    ((6, 4, 10, true, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((6, 4, 10, false, 0.1, 0.1, 0.1, 0.2, 0.2), SOCExpPSDOptimizer),
    ((6, 4, 10, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((12, 8, 20, true, 0.1, 0.1, 0.1, 0.2, 0.2), SOCExpPSDOptimizer),
    ((12, 8, 20, true, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((12, 8, 20, false, 0.1, 0.1, 0.1, 0.2, 0.2), SOCExpPSDOptimizer),
    ((12, 8, 20, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ]
return (MatrixRegressionJuMP, insts)
