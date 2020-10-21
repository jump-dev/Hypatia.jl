
nearestpsd_insts(use_nat::Bool) = [
    [(side, use_completable, false, use_nat) for side in vcat(30, 50:50:800)] # includes compile run
    for use_completable in (false, true)
    ]

insts = Dict()
insts["nat"] = (nothing, nearestpsd_insts(true))
insts["ext"] = (StandardConeOptimizer, nearestpsd_insts(false))
return (NearestPSDJuMP, insts)
