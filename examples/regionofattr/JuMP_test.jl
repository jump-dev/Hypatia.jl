
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
    ((10, true),),
    ((10, false), SOCExpPSDOptimizer),
    ]
insts["various"] = vcat(insts["fast"], insts["slow"])
return (RegionOfAttrJuMP, insts)
