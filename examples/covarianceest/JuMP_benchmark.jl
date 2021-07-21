
covarianceest_insts(ext::MatSpecExt) = [
    [(d, ext)
    for d in vcat(3, 20:20:220)] # includes compile run
    ]

insts = OrderedDict()
insts["logdet"] = (nothing, covarianceest_insts(MatLogdetCone()))
insts["sepspec"] = (nothing, covarianceest_insts(MatNegLog()))
insts["direct"] = (nothing, covarianceest_insts(MatNegLogDirect()))
return (CovarianceEstJuMP, insts)
