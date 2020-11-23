
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
    ((5, 10),),
    ((5, 6), SOCExpPSDOptimizer),
    ((10, 20, 20, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((10, 20, 20, false, 0, 0.1, 0.1, 0, 0),),
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
return (MatrixRegressionJuMP, insts)
