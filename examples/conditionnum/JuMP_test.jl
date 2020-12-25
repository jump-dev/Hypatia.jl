
relaxed_tols = (default_tol_relax = 100,)
insts = Dict()
insts["minimal"] = [
    ((3, 2, true),),
    ((3, 2, false),),
    ]
insts["fast"] = [
    ((3, 4, true), nothing),
    ((3, 4, false), nothing),
    ((10, 15, true), nothing),
    ((10, 15, false), nothing),
    ((20, 10, true), nothing, relaxed_tols),
    ((100, 40, false), nothing, relaxed_tols),
    ]
insts["slow"] = [
    ((100, 10, true), nothing),
    ((100, 40, true), nothing),
    ]
return (ConditionNumJuMP, insts)
