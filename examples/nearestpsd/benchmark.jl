
nearestpsd_instances(use_nat::Bool) = [
    [(side, use_completable, false, use_nat) for side in vcat(40, 50:50:800)] # includes compile run
    for use_completable in (false, true)
    ]

instances[NearestPSDJuMP]["nat"] = (nothing, nearestpsd_instances(true))
instances[NearestPSDJuMP]["ext"] = (StandardConeOptimizer, nearestpsd_instances(false))
