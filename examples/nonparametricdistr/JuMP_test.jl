
relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [
    ((2, 2, false),),
    ((4, 1, false),),
    ((4, 3, false),),
    ((4, 2, true), nothing, (default_tol_relax = 100,)),
    ]
insts["fast"] = [
    ((1000, 1, false),),
    ((200, 1, true), nothing, relaxed_tols),
    ((1000, 2, false),),
    ((200, 2, true), nothing, relaxed_tols),
    ((1000, 3, false), nothing, relaxed_tols),
    ]
insts["various"] = [
    ((1000, 1, false),),
    ((7000, 1, false),),
    ((500, 1, true), nothing, relaxed_tols),
    ((1000, 1, true), nothing, relaxed_tols),
    ((1000, 2, false),),
    ((500, 2, true), nothing, relaxed_tols),
    ((2000, 3, false), nothing, relaxed_tols),
    ((100, 3, true), nothing, relaxed_tols),
    ((400, 3, true), nothing, relaxed_tols),
    ((4000, 3, false), nothing, relaxed_tols),
    ]
insts["natvext"] = [
    # ((5000, 1, true),), # 28 iterations and 341.91 seconds
    # ((7000, 1, true),), # 27 iterations and 832.922 seconds
    # ((8000, 1, true),), # 27 iterations and 1218.224 seconds
    # ((15000, 1, false),), # 27 iterations and 857.543 seconds
    ((17000, 1, false),),
    ]
return (NonparametricDistrJuMP, insts)
