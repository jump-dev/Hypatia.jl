
nonparametricdistr_insts(ext::VecSpecExt) = [
    [(d, [ext])
    for d in vcat(10, 2000:2000:22000)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(
    nonparametricdistr_insts(VecNegLog()),
    nonparametricdistr_insts(VecNegEntropy()),
    nonparametricdistr_insts(VecPower12(1.5)),
    ))
insts["ext"] = (nothing, vcat(
    nonparametricdistr_insts(VecNegLogEF()),
    nonparametricdistr_insts(VecNegEntropyEF()),
    nonparametricdistr_insts(VecPower12EF(1.5)),
    ))
return (NonparametricDistrJuMP, insts)
