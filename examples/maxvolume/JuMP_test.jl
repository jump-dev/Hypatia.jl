
relaxed_tols = (default_tol_relax = 100,)
insts = OrderedDict()
insts["minimal"] = [
    ((2, true, false),),
    ((2, false, true),),
    ((2, false, true), :SOCExpPSD, relaxed_tols),
    ]
insts["fast"] = [
    ((10, true, false),),
    ((10, false, true), nothing, relaxed_tols),
    ((10, false, true), :SOCExpPSD),
    ((100, true, false),),
    ((100, false, true), nothing, relaxed_tols),
    ((100, false, true), :SOCExpPSD, relaxed_tols),
    ((1000, true, false),),
    ]
insts["various"] = [
    ((500, true, false),),
    ((500, false, true),),
    ((500, true, false), :SOCExpPSD),
    ((500, false, true), :SOCExpPSD),
    ((1000, true, false),),
    ((1000, false, true),),
    ((1000, true, false), :SOCExpPSD),
    ((1000, false, true), :SOCExpPSD, relaxed_tols),
    ((2000, true, false),),
    ((2000, false, true),),
    ((2000, true, false), :SOCExpPSD),
    ]
return (MaxVolumeJuMP, insts)
