
insts = Dict()
insts["minimal"] = [
    ((2, true),),
    ((2, false),),
    ]
insts["fast"] = [
    ((20, true),),
    ((20, false),),
    ((50, true),),
    ((50, false),),
    ]
insts["slow"] = [
    ((500, true),),
    ((500, false),),
    ]
return (StabilityNumber, insts)
