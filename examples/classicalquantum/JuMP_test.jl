
insts = OrderedDict()
insts["minimal"] = [
    ((3, false, false),),
    ((3, false, true),),
    ((3, true, false),),
    ]
insts["fast"] = [
    ((20, false, false),),
    ((20, true, false),),
    ((8, false, true),),
    ((50, false, false),),
    ((50, true, false),),
    ((12, false, true),),
    ]
insts["various"] = [
    ((100, false, false),),
    ((100, true, false),),
    ((12, false, true),),
    ((200, false, false),),
    ((200, true, false),),
    ((15, false, true), nothing, (default_tol_relax = 100,)),
    ((300, false, false),),
    ((250, true, false),),
    ((17, false, true), nothing, (default_tol_relax = 1000,)),
    ]
return (ClassicalQuantum, insts)
