
insts = Dict()
insts["minimal"] = [
    ((2, 4, true, true),),
    ((2, 4, true, false),),
    ((2, 2, false, true),),
    ((2, 2, false, false),),
    ]
insts["fast"] = [
    ((3, 6, true, true),),
    ((3, 6, true, false),),
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
insts["various"] = [
    ((6, 6, true, true),),
    ((6, 6, true, false),),
    ((6, 6, false, true),),
    ((6, 6, false, false),),
    ((12, 12, true, true),),
    ((12, 12, true, false),),
    ((12, 12, false, true),),
    ((12, 12, false, false),),
    ((24, 24, true, true),),
    ((24, 24, true, false),),
    ((24, 24, false, true),),
    ((24, 24, false, false),),
    ]
return (LyapunovStabilityJuMP, insts)
