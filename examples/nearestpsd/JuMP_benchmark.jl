
nearestpsd_insts(use_nat::Bool) = [
    [(side, use_completable, false, use_nat, true)
    for side in vcat(30, 50:50:200, 300:100:800)] # includes compile run
    for use_completable in (false, true)
    ]

insts = OrderedDict()
insts["nat"] = (nothing, nearestpsd_insts(true))
insts["ext"] = (:SOCExpPSD, nearestpsd_insts(false))
return (NearestPSDJuMP, insts)
