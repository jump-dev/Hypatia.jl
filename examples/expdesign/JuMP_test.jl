
relaxed_tols = (default_tol_relax = 100,)
insts = Dict()
insts["minimal"] = [
    ((2, 3, 4, 2, true, false, false), nothing, relaxed_tols),
    ((2, 3, 4, 2, true, false, false), SOCExpPSDOptimizer),
    ((2, 3, 4, 2, false, true, false),),
    ((2, 3, 4, 2, false, true, false), ExpPSDOptimizer),
    ((2, 3, 4, 2, false, false, true),),
    ((2, 3, 4, 2, false, false, true), ExpPSDOptimizer),
    ]
insts["fast"] = [
    ((3, 5, 7, 2, true, false, false),),
    ((3, 5, 7, 2, true, false, false), SOCExpPSDOptimizer),
    ((3, 5, 7, 2, false, true, false),),
    ((3, 5, 7, 2, false, true, false), ExpPSDOptimizer),
    ((3, 5, 7, 2, false, false, true),),
    ((3, 5, 7, 2, false, false, true), ExpPSDOptimizer),
    ((5, 15, 25, 5, true, false, false),),
    ((5, 15, 25, 5, false, true, false),),
    ((5, 15, 25, 5, false, false, true),),
    ((10, 30, 50, 5, true, false, false),),
    ((10, 30, 50, 5, false, true, false),),
    ((10, 30, 50, 5, false, false, true),),
    ((25, 75, 125, 10, true, false, false),),
    ((25, 75, 125, 10, false, true, false),),
    ((25, 75, 125, 10, false, false, true),),
    ]
insts["slow"] = [
    ((25, 75, 125, 10, true, false, false), SOCExpPSDOptimizer),
    ((25, 75, 125, 10, false, true, false), ExpPSDOptimizer),
    ((25, 75, 125, 10, false, false, true), ExpPSDOptimizer),
    ((100, 200, 200, 10, true, false, false),),
    ((100, 200, 200, 10, false, true, false),),
    ((100, 200, 200, 10, false, false, true),),
    ]
return (ExpDesignJuMP, insts)
