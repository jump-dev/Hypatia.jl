
relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [
    ((2, 2, false),),
    ((4, 1, false),),
    ((4, 2, true),),
    ((4, 3, false),),
    ((4, 3, true),),
    ]
insts["fast"] = [
    ((1000, 2, false),),
    ((120, 2, true), nothing, relaxed_tols),
    ((700, 5, false),),
    ((100, 5, true), nothing, relaxed_tols),
    ]
insts["various"] = [
    ((1500, 2, false),),
    ((500, 2, true), nothing, relaxed_tols),
    ((1000, 5, false),),
    ((200, 5, true), nothing, relaxed_tols),
    ((3000, 2, false),),
    ((1000, 2, true), nothing, relaxed_tols),
    ((2000, 5, false),),
    ((400, 5, true), nothing, relaxed_tols),
    ((4000, 3, false),),
    ((6000, 3, false),),
    ]
return (NonparametricDistrJuMP, insts)
