
covarianceest_insts(ext::MatSpecExt) = [
    [(d, ext)
    for d in vcat(3, 10:10:40, 50:25:175)] # includes compile run
    ]

insts = OrderedDict()
insts["logdet"] = (nothing, covarianceest_insts(MatLogdetCone()))
insts["sepspec"] = (nothing, covarianceest_insts(MatNegLog()))
insts["direct"] = (nothing, covarianceest_insts(MatNegLogDirect()))
insts["eigord"] = (nothing, covarianceest_insts(MatNegLogEigOrd()))
return (CovarianceEstJuMP, insts)
