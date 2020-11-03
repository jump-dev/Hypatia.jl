
insts = Dict()
insts["minimal"] = [
    ((5, 1, 2, true),),
    ((5, 1, 2, false),),
    ((5, 1, 2, false), ExpPSDOptimizer),
    ((5, 1, 2, false), SOCExpPSDOptimizer),
    ((:iris, 2, true),),
    ]
insts["fast"] = [
    ((10, 1, 5, true),),
    ((10, 1, 10, true),),
    ((10, 1, 10, false),),
    ((10, 1, 10, false), ExpPSDOptimizer),
    ((10, 1, 10, false), SOCExpPSDOptimizer),
    ((100, 1, 250, true),),
    ((100, 2, 5, true),),
    ((200, 2, 20, true),),
    ((50, 3, 2, true),),
    ((50, 3, 2, false),),
    ((50, 3, 2, false), ExpPSDOptimizer),
    ((50, 3, 2, false), SOCExpPSDOptimizer),
    ((50, 3, 4, true),),
    ((500, 3, 14, true),),
    ((50, 4, 2, true),),
    ((100, 8, 2, true),),
    ((100, 8, 2, false),),
    ((250, 4, 4, true),),
    ((250, 4, 4, false),),
    ((200, 32, 2, false),),
    ((:iris, 4, true),),
    ((:iris, 5, true),),
    ((:iris, 6, true),),
    ((:iris, 4, false),),
    ((:cancer, 4, true),),
    ]
insts["slow"] = [
    ((500, 2, 60, true),),
    ((1000, 3, 20, true),),
    ((200, 4, 4, false),),
    ((500, 4, 6, true),),
    ((500, 4, 6, false),),
    ]
return (DensityEstJuMP, insts)
