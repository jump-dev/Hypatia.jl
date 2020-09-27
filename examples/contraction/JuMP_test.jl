
insts = Dict()
insts["minimal"] = [
    ((0.77, 4, 1e-3, true, true),),
    ((0.77, 4, 1e-3, false, true), StandardConeOptimizer),
    ((0.85, 4, 1e-3, true, false),),
    ((0.85, 4, 1e-3, false, false), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((0.77, 6, 1e-3, true, true),),
    ((0.77, 6, 1e-3, false, true), StandardConeOptimizer),
    ((0.85, 6, 1e-3, true, false),),
    ((0.85, 6, 1e-3, false, false), StandardConeOptimizer),
    ]
insts["slow"] = Tuple[]
return (ContractionJuMP, insts)
