
insts = Dict()
insts["minimal"] = [
    ((2, 3, true, true, true, true),),
    ((2, 3, false, true, true, true),),
    ((2, 3, true, false, true, true),),
    ((2, 3, false, false, true, true),),
    ((2, 3, true, true, false, true),),
    ((2, 3, false, false, false, true),),
    ((2, 3, true, true, false, false),),
    ((2, 3, false, false, false, false),),
    ]
insts["fast"] = [
    ((12, 24, true, true, true, true),),
    ((12, 24, false, true, true, true),),
    ((12, 24, true, false, true, true),),
    ((12, 24, false, false, true, true),),
    ((12, 24, true, true, false, true),),
    ((12, 24, false, false, false, true),),
    ((12, 24, true, true, false, false),),
    ((12, 24, false, false, false, false),),
    ]
insts["slow"] = [
    ((14, 140, true, true, true, true),),
    ((14, 140, true, true, false, true),),
    ((14, 140, true, true, true, false),),
    ((14, 140, true, true, false, false),),
    ((18, 180, true, true, true, true),),
    ((18, 180, true, true, false, true),),
    ((18, 180, true, true, true, false),),
    ((18, 180, true, true, false, false),),
    ]
insts["various"] = [
    ((5, 100, true, true, true, true),),
    ((5, 100, true, true, false, true),),
    ((10, 200, true, true, true, true),),
    ((10, 200, true, true, false, true),),
    ]
return (MatrixCompletionNative, insts)
