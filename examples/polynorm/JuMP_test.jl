
relaxed_tols = (default_tol_relax = 100,)
insts = OrderedDict()
insts["minimal"] = [
    ((1, 1, 1, 2, true, true, false), nothing, relaxed_tols),
    ((1, 1, 1, 2, true, false, true),),
    ((1, 1, 1, 2, false, true, false),),
    ((1, 1, 1, 2, false, false, true),),
    ((1, 1, 1, 2, false, false, false),),
    ]
insts["fast"] = [
    ((2, 2, 2, 2, true, true, false),),
    ((2, 2, 2, 2, true, false, true),),
    ((2, 2, 2, 2, false, true, false),),
    ((2, 2, 2, 2, false, false, true),),
    ((2, 2, 2, 2, false, false, false),),
    ((4, 2, 3, 5, true, true, false),),
    ((4, 2, 3, 5, true, false, true),),
    ]
insts["various"] = vcat(insts["fast"], [
    ((4, 2, 3, 5, false, true, false),),
    ((4, 2, 3, 5, false, false, true),),
    ((4, 2, 3, 5, false, false, false),),
    ])
return (PolyNormJuMP, insts)
