
insts = Dict()
insts["minimal"] = [
    ((1, 1, 1, 2, true, true),),
    ((1, 1, 1, 2, true, false),),
    ((1, 1, 1, 2, false, true),),
    ((1, 1, 1, 2, false, false),),
    ]
insts["fast"] = [
    ((2, 2, 1, 2, true, true),),
    ((2, 2, 1, 2, false, false),),
    ((2, 1, 1, 3, true, false),),
    ((2, 1, 1, 3, false, true),),
    ((4, 3, 2, 10, true, true),),
    ((4, 3, 2, 10, true, false),),
    ]
insts["slow"] = [
    ((4, 3, 2, 10, false, true),),
    ((4, 3, 2, 10, false, false),),
    ]
return (PolyNormJuMP, insts)
