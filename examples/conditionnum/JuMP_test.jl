
relaxed_tols = (default_tol_relax = 100,)
insts = Dict()
insts["minimal"] = [
    ((3, 2, true),),
    ((3, 2, false),),
    ]
insts["fast"] = [
    ((3, 4, true),),
    ((3, 4, false),),
    ((10, 15, true),),
    ((10, 15, false),),
    ((20, 10, true), nothing, relaxed_tols),
    ((100, 40, false), nothing, relaxed_tols),
    ]
insts["slow"] = [
    ((100, 10, true),),
    ((100, 40, true),),
    ]
insts["various"] = [
    ((3, 4, true), nothing, relaxed_tols),
    ((3, 4, false), nothing, relaxed_tols),
    ((10, 15, true), nothing, relaxed_tols),
    ((10, 15, false), nothing, relaxed_tols),
    ((100, 10, true), nothing, relaxed_tols),
    ((100, 10, false), nothing, relaxed_tols),
    ]
return (ConditionNumJuMP, insts)
