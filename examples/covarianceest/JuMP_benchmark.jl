
covarianceest_insts(ext::MatSpecExt) = [[(d, true, ext) # complex
                                         for d in vcat(3, 25:25:250)]]

insts = OrderedDict()
insts["logdet"] = (nothing, covarianceest_insts(MatLogdetCone()))
insts["sepspec"] = (nothing, covarianceest_insts(MatNegLog()))
insts["direct"] = (nothing, covarianceest_insts(MatNegLogDirect()))
return (CovarianceEstJuMP, insts)
