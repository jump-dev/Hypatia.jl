
relaxed_tols = (default_tol_relax = 100,)
insts = Dict()
insts["minimal"] = [
    ((1, 1, 1, false, false, true),),
    ((1, 1, 1, false, true, false),),
    ((1, 1, 1, true, false, false),),
    ]
insts["fast"] = [
    ((3, 1, 3, false, false, true),),
    ((3, 1, 3, false, true, false),),
    ((3, 1, 3, true, false, false),),
    ((1, 2, 3, false, false, true),),
    ((1, 2, 3, false, true, false),),
    ((1, 2, 3, true, false, false),),
    ]
insts["slow"] = [
    ((4, 2, 5, false, true, false),),
    ((4, 2, 5, true, false, false),),
    ]
insts["various"] = [
    ((4, 2, 5, false, false, true),),
    ((4, 2, 5, false, true, false),),
    ((4, 2, 5, true, false, false),),
    ((4, 2, 10, false, true, false), nothing, relaxed_tols),
    ((4, 2, 10, true, false, false),),
    ((4, 4, 5, false, true, false), nothing, relaxed_tols),
    ((4, 4, 5, true, false, false),),
    ]
return (NearestPolyMatJuMP, insts)
