
relaxed_tols = (default_tol_relax = 100,)
insts = Dict()
insts["minimal"] = [
    ((2, true, false),),
    ((2, true, true), nothing, relaxed_tols),
    ((2, false, true),),
    ((2, false, true), SOCExpPSDOptimizer, relaxed_tols),
    ]
insts["fast"] = [
    ((10, true, false),),
    ((10, false, true), nothing, relaxed_tols),
    ((10, false, true), SOCExpPSDOptimizer),
    ((10, true, true), nothing, relaxed_tols),
    ((100, true, false),),
    ((100, false, true), nothing, relaxed_tols),
    ((100, false, true), SOCExpPSDOptimizer, relaxed_tols),
    ((100, true, true), nothing, relaxed_tols),
    ((1000, true, false),),
    ((1000, true, true), nothing, relaxed_tols), # with bridges extended formulation will need to go into slow list
    ]
insts["slow"] = [
    ((1000, false, true), SOCExpPSDOptimizer, relaxed_tols),
    ((2000, true, false),),
    ((2000, false, true), nothing, relaxed_tols),
    ((2000, true, true), nothing, relaxed_tols),
    ]
return (MaxVolumeJuMP, insts)
