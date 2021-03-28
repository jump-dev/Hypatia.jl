
relaxed_tols = (default_tol_relax = 1000,)
insts = Dict()
insts["minimal"] = [
    ((5, 2, 2, true, true, true),),
    ((5, 1, 2, false, true, true),),
    ((5, 1, 2, false, false, false),),
    ((5, 1, 2, true, false, false), ExpPSDOptimizer),
    ((5, 1, 2, true, false, false), SOCExpPSDOptimizer),
    ((:iris, 2, true, true, true),),
    ]
insts["fast"] = [
    ((10, 1, 10, true, false, false), ExpPSDOptimizer),
    ((10, 1, 10, true, false, false), SOCExpPSDOptimizer),
    ((100, 1, 250, true, true, true),),
    ((100, 2, 5, false, true, true),),
    ((200, 2, 20, true, true, true),),
    ((50, 3, 2, true, true, true),),
    ((50, 3, 2, false, false, true),),
    ((50, 3, 2, true, false, true), ExpPSDOptimizer),
    ((50, 3, 2, true, false, true), SOCExpPSDOptimizer),
    ((50, 3, 4, true, true, true),),
    ((500, 3, 14, true, true, true),),
    ((100, 8, 2, true, true, true),),
    ((250, 4, 4, true, true, true),),
    ((250, 4, 4, false, false, true),),
    ((200, 32, 2, true, false, true),),
    ((:iris, 4, false, false, false),),
    ((:iris, 5, true, false, false),),
    ((:iris, 6, true, true, true),),
    ((:cancer, 4, true, true, true),),
    ]
insts["slow"] = [
    ((500, 2, 50, true, true, true),),
    ((1000, 3, 16, true, true, true),),
    ((200, 6, 4, false, true, true),),
    ((500, 6, 6, true, true, true),),
    ((500, 4, 6, true, false, true),),
    ]
insts["various"] = [
    ((50, 2, 16, true, false, false),),
    ((50, 2, 16, true, false, false), ExpPSDOptimizer),
    ((50, 2, 16, true, false, false), SOCExpPSDOptimizer),
    ((50, 2, 16, false, false, false),),
    ((50, 2, 16, false, false, true),),
    ((50, 8, 4, true, false, false),),
    ((50, 8, 4, true, false, false), ExpPSDOptimizer),
    ((50, 8, 4, true, false, false), SOCExpPSDOptimizer),
    ((50, 8, 4, false, false, false),),
    ((50, 8, 4, false, false, true),),
    ((50, 32, 2, true, false, false),),
    ((50, 32, 2, true, false, false), ExpPSDOptimizer),
    ((50, 32, 2, true, false, false), SOCExpPSDOptimizer),
    ((50, 32, 2, false, false, false), nothing, relaxed_tols),
    ((50, 32, 2, false, false, true), nothing, relaxed_tols),
    ((:iris, 6, true, true, true),),
    ((:cancer, 4, true, true, true),),
    ]
return (DensityEstJuMP, insts)
