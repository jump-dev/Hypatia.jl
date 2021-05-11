
insts = Dict()
insts["minimal"] = [
    ((1, 2, 2, 2, true),),
    ((1, 2, 2, 2, false),),
    ]
insts["fast"] = [
    ((2, 2, 3, 2, true),),
    ((2, 2, 3, 2, false),),
    ((3, 3, 3, 3, true),),
    ((3, 3, 3, 3, false),),
    ((3, 3, 5, 4, true),),
    ((5, 2, 5, 3, true),),
    ((1, 30, 2, 30, true),),
    ((1, 30, 2, 30, false),),
    ((10, 1, 3, 1, true),),
    ((10, 1, 3, 1, false),),
    ]
insts["slow"] = [
    ((3, 3, 5, 4, false),),
    ((5, 2, 5, 3, false),),
    ((4, 6, 4, 5, true),),
    ((4, 6, 4, 5, false),),
    ((2, 30, 4, 30, true),),
    ((2, 30, 4, 30, false),),
    ]
insts["various"] = Tuple[]
return (EnvelopeNative, insts)
