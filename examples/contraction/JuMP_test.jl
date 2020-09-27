
insts = Dict()
insts["minimal"] = [
    ((0.7, 4, 1e-3, true, true),),
    ((0.7, 4, 1e-3, false, true), StandardConeOptimizer),
    ((1.0, 4, 1e-3, true, false),),
    ((1.0, 4, 1e-3, false, false), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((1.0, 2, 1e-3, true, false),),
    ((1.0, 2, 1e-3, false, false), StandardConeOptimizer),
    ((2.0, 6, 1e-3, true, false),),
    ((2.0, 6, 1e-3, false, false), StandardConeOptimizer),
    ]
insts["slow"] = Tuple[]
return (ContractionJuMP, insts)
