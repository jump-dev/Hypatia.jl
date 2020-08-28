
insts = Dict()
insts["minimal"] = [
    ((3, 2, true, 0, false),),
    ((3, 2, false, 0, false),),
    ((3, 2, true, 10, false),),
    ((3, 2, false, 10, false),),
    ]
insts["fast"] = [
    ((5, 3, true, 0, false),),
    ((5, 3, false, 0, false),),
    ((5, 3, true, 10, false),),
    ((5, 3, false, 10, false),),
    ((30, 10, true, 0, false),),
    ((30, 10, false, 0, false),),
    ((30, 10, true, 10, false),),
    ((30, 10, false, 10, false),),
    ]
insts["slow"] = Tuple[]
insts["linops"] = [
    ((5, 3, true, 0, true),),
    ((5, 3, false, 0, true),),
    ((5, 3, true, 10, true),),
    ((5, 3, false, 10, true),),
    ((30, 10, true, 0, true),),
    ((30, 10, false, 0, true),),
    ((30, 10, true, 10, true),),
    ((30, 10, false, 10, true),),
    ]
return (SparsePCANative, insts)
