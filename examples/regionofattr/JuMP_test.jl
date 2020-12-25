
# NOTE an extender is needed if use_wsos = false
insts = Dict()
insts["minimal"] = [
    ((4, true),),
    ((4, false), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    ((6, true),),
    ((6, false), SOCExpPSDOptimizer),
    ((8, true),),
    ]
insts["slow"] = [
    ((8, false), SOCExpPSDOptimizer),
    ]
return (RegionOfAttrJuMP, insts)
