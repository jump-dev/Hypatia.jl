
covarianceest_ds = vcat(3, 10:10:40, 50:25:175) # includes compile run
insts = OrderedDict()
insts["nat"] = (nothing, [[(d, MatNegLog()) for d in covarianceest_ds]])
insts["extdirect"] = (nothing, [[(d, MatNegLogDirect()) for d in covarianceest_ds]])
insts["extord"] = (nothing, [[(d, MatNegLogEigOrd()) for d in covarianceest_ds]])
insts["logdet"] = (nothing, [[(d, nothing) for d in covarianceest_ds]])
return (CovarianceEstJuMP, insts)
