
insts = Dict()
insts["minimal"] = [
    ((2, 3, true, true),),
    ((2, 3, true, false),),
    ((2, 2, false, true),),
    ((2, 2, false, false),),
    ]
insts["fast"] = [
    ((5, 6, true, true),),
    ((5, 6, true, false),),
    ((5, 5, false, true),),
    ((5, 5, false, false),),
    ((10, 20, true, true),),
    ((10, 20, true, false),),
    ((15, 15, false, true),),
    ((15, 15, false, false),),
    ((30, 30, false, false),),
    ((30, 30, false, true),),
    ]
insts["slow"] = [
    ((50, 50, false, false),),
    ((50, 50, false, true),),
    ]
return (LyapunovStabilityJuMP, insts)
