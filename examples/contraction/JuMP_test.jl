
insts = Dict()
insts["minimal"] = [
    ((0.85, 2, 1e-3, true, false),),
    ((0.85, 2, 1e-3, false, false),),
    ((0.85, 2, 1e-3, true, true),),
    ((0.85, 2, 1e-3, false, true),),
    ]
insts["fast"] = [
    ((0.77, 4, 1e-3, true, true),),
    ((0.77, 4, 1e-3, false, true),),
    ((0.85, 4, 1e-3, true, false),),
    ((0.85, 4, 1e-3, false, false),),
    ]
insts["slow"] = Tuple[]
return (ContractionJuMP, insts)
