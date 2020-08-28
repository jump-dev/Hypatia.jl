
insts = Dict()
insts["minimal"] = [
    ((1, 2, false),),
    ((1, 2, false), StandardConeOptimizer),
    ((2, 2, true),),
    ((2, 2, true), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((1, 10, true),),
    ((1, 10, true), StandardConeOptimizer),
    ((1, 15, false),),
    ((1, 15, false), StandardConeOptimizer),
    ((2, 3, true),),
    ((2, 3, true), StandardConeOptimizer),
    ((2, 3, false),),
    ((2, 3, false), StandardConeOptimizer),
    ((2, 6, true),),
    ((2, 5, true), StandardConeOptimizer),
    ((2, 7, false),),
    ((2, 6, false), StandardConeOptimizer),
    ((3, 2, true),),
    ((3, 2, false),),
    ((3, 4, true),),
    ((3, 4, false),),
    ((7, 2, true),),
    ((7, 2, true), StandardConeOptimizer),
    ((7, 2, false),),
    ((7, 2, false), StandardConeOptimizer),
    ]
insts["slow"] = [
    ((1, 20, false),),
    ((2, 3, false),),
    ((2, 10, false),),
    ((2, 8, false), StandardConeOptimizer),
    ((3, 4, true), StandardConeOptimizer),
    ((3, 4, false), StandardConeOptimizer),
    ((3, 5, true),),
    ((3, 5, false),),
    ((6, 3, true),),
    ((6, 3, false),),
    ]
return (CentralPolyMatJuMP, insts)
