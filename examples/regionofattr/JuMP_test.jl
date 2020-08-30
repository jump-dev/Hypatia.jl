
relaxed_tols = (tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, tol_feas = 1e-6)
insts = Dict()
insts["minimal"] = [
    ((4, true),),
    ((4, false), nothing, relaxed_tols),
    ]
insts["fast"] = [
    ((6, true),),
    ((6, false),),
    ((8, true),),
    ]
insts["slow"] = [
    ((8, false),),
    ]
return (RegionOfAttrJuMP, insts)
