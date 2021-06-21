
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
insts["natvext"] = [
    # ((400, false, false),), # good, 1020.576 seconds
    # ((500, false, false),), # close to converging after 2040.447 seconds
    # ((20, false, true), nothing, (default_tol_relax = 1000,)), # 53.465 seconds
    # ((50, false, true), nothing, (default_tol_relax = 1000,)), # killed before solving but didn't investigate
    # ((25, false, true), nothing, (default_tol_relax = 1000,)), # 312.469 seconds
    # ((30, false, true), nothing, (default_tol_relax = 1000,)), # 1477.232 seconds
    ((10, false, true),),
    ((20, false, true),),
    ((30, false, true),),
    # ((40, false, true),), # time limit
    # ((50, false, true),), # very dead
    #
    ((10, false, false),),
    ((20, false, false),),
    ((30, false, false),),
    ((40, false, false),),
    ((50, false, false),),
    #
    ((100, false, false),),
    ((200, false, false),),
    ((300, false, false),),
    ((400, false, false),),
    ((500, false, false),),
    ]
return (ClassicalQuantum, insts)
