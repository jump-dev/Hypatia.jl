
insts = Dict()
insts["minimal"] = [
    ((0.7, 4, 1e-3, true, true),),
    ((0.7, 4, 1e-3, false, true), SOCExpPSDOptimizer),
    ((1.0, 4, 1e-3, true, false),),
    ((1.0, 4, 1e-3, false, false), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    ((1.0, 2, 1e-3, true, false),),
    ((1.0, 2, 1e-3, false, false), SOCExpPSDOptimizer),
    ((2.0, 6, 1e-3, true, false),),
    ((2.0, 6, 1e-3, false, false), SOCExpPSDOptimizer),
    ]
insts["slow"] = Tuple[]
insts["various"] = vcat(insts["minimal"], insts["fast"])
return (ContractionJuMP, insts)
