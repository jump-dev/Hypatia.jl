
insts = OrderedDict()
insts["minimal"] = [
    ((5, 2, 2, true, true, true),),
    ((5, 1, 2, false, true, true),),
    ((5, 1, 2, false, true, false),),
    ((5, 1, 2, true, false, false), :ExpPSD),
    ((5, 1, 2, true, false, false), :SOCExpPSD),
    ((:iris, 2, true, false, true),),
    ]
insts["fast"] = [
    ((10, 1, 10, true, false, false), :ExpPSD),
    ((10, 1, 10, true, false, false), :SOCExpPSD),
    ((100, 1, 250, true, true, true),),
    ((100, 2, 5, false, true, true),),
    ((200, 2, 20, true, true, true),),
    ((50, 3, 2, true, true, true),),
    ((50, 3, 2, false, false, true),),
    ((50, 3, 2, true, false, true), :ExpPSD),
    ((50, 3, 2, true, false, true), :SOCExpPSD),
    ((50, 3, 4, true, true, true),),
    ((500, 3, 14, true, true, true),),
    ((100, 8, 2, true, true, true),),
    ((250, 4, 4, true, true, true),),
    ((250, 4, 4, false, false, true),),
    ((200, 32, 2, true, false, true),),
    ((:iris, 4, false, false, false),),
    ((:iris, 5, true, false, false),),
    ((:iris, 6, true, true, true),),
    ((:cancer, 4, true, true, true),),
    ]
insts["various"] = [
    ((50, 2, 16, true, false, false),),
    ((50, 2, 16, true, false, false), :ExpPSD),
    ((50, 2, 16, true, false, false), :SOCExpPSD),
    ((50, 2, 16, false, false, false),),
    ((50, 2, 16, false, false, true),),
    ((50, 8, 4, true, false, false),),
    ((50, 8, 4, true, false, false), :ExpPSD),
    ((50, 8, 4, true, false, false), :SOCExpPSD),
    ((50, 8, 4, false, false, false),),
    ((50, 8, 4, false, false, true),),
    ((50, 32, 2, true, false, false),),
    ((50, 32, 2, true, false, false), :ExpPSD),
    ((50, 32, 2, true, false, false), :SOCExpPSD),
    ((:iris, 6, true, true, true),),
    ((:cancer, 4, true, true, true),),
    ]
return (DensityEstJuMP, insts)
