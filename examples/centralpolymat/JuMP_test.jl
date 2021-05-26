
insts = OrderedDict()
insts["minimal"] = [
    ((1, 2, false),),
    ((1, 2, false), :SOCExpPSD),
    ((2, 2, true),),
    ((2, 2, true), :ExpPSD),
    ]
insts["fast"] = [
    ((1, 10, true),),
    ((1, 10, true), :SOCExpPSD),
    ((1, 15, false),),
    ((1, 15, false), :ExpPSD),
    ((2, 3, true),),
    ((2, 3, true), :SOCExpPSD),
    ((2, 3, false),),
    ((2, 3, false), :ExpPSD),
    ((2, 6, true),),
    ((2, 5, true), :SOCExpPSD),
    ((2, 7, false),),
    ((2, 6, false), :SOCExpPSD),
    ((3, 2, true),),
    ((3, 2, false),),
    ((3, 4, true),),
    ((3, 4, false),),
    ((7, 2, true),),
    ((7, 2, true), :ExpPSD),
    ((7, 2, false),),
    ((7, 2, false), :SOCExpPSD),
    ]
insts["various"] = [
    ((2, 5, true),),
    ((2, 5, true), :ExpPSD),
    ((2, 5, false),),
    ((2, 5, false), :SOCExpPSD),
    ((2, 5, false), :ExpPSD),
    ((2, 10, true),),
    ((2, 10, true), :ExpPSD),
    ((2, 10, false),),
    ((2, 10, false), :SOCExpPSD),
    ((2, 10, false), :ExpPSD),
    ((6, 3, true),),
    ((6, 3, false),),
    ((8, 3, true),),
    ((8, 3, false),),
    ]
return (CentralPolyMatJuMP, insts)
