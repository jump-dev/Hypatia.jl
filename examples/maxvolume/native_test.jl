
insts = Dict()
insts["minimal"] = [
    ((2, true, false, false),),
    ((3, false, true, false),),
    ((2, false, false, true),),
    ]
insts["fast"] = [
    ((10, true, false, false),),
    ((10, false, true, false),),
    ((10, false, false, true),),
    ((100, true, false, false),),
    ((100, false, true, false), (default_tol_relax = 100,)),
    ((100, false, false, true),),
    ((1000, true, false, false),),
    ]
insts["slow"] = [
    ((1000, false, true, false),),
    ((1000, false, false, true),),
    ((1500, true, false, false),),
    ((1500, false, true, false),),
    ((1500, false, false, true),),
    ]
insts["various"] = [
    ((750, true, false, false),),
    ((750, false, true, false),),
    ((750, false, false, true),),
    ((1500, true, false, false),),
    ((1500, false, true, false),),
    ((1500, false, false, true),),
    ]
return (MaxVolumeNative, insts)
