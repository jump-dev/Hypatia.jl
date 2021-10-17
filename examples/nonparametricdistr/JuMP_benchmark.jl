
nonparametricdistr_insts(ext::VecSpecExt) = [
    [(d, ext)
    for d in vcat(10, 500, 1000, 2500, 5000:5000:25000)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(nonparametricdistr_insts.([
    VecNegGeom(),
    VecNegLog(),
    VecNegSqrt(),
    VecNegEntropy(),
    ])...))
insts["natlog"] = (nothing, nonparametricdistr_insts(
    VecLogCone()))
insts["vecext"] = (nothing, vcat(nonparametricdistr_insts.([
    VecNegGeomEFExp(),
    VecNegLogEF(),
    VecNegSqrtEF(),
    VecNegEntropyEF(),
    ])...))
return (NonparametricDistrJuMP, insts)
