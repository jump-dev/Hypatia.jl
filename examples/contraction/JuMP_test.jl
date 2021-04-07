
relaxed_infeas_tols = (tol_infeas = 1e-10,)
insts = Dict()
insts["minimal"] = [
    ((0.7, 4, 1e-3, true, true),),
    ((0.7, 4, 1e-3, false, true), SOCExpPSDOptimizer),
    ((2.0, 4, 1e-3, true, false), nothing, relaxed_infeas_tols),
    ((2.0, 4, 1e-3, false, false), SOCExpPSDOptimizer, relaxed_infeas_tols),
    ]
insts["fast"] = [
    ((1.0, 2, 1e-3, true, false),),
    ((1.0, 2, 1e-3, false, false), SOCExpPSDOptimizer),
    ((4.0, 2, 1e-3, true, false),),
    ((4.0, 2, 1e-3, false, false), SOCExpPSDOptimizer),
    ]
insts["slow"] = Tuple[]
insts["various"] = vcat(insts["minimal"], insts["fast"])
return (ContractionJuMP, insts)
