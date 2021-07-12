
nonparametricdistr_insts(exts::Vector{VecSpecExt}) = [
    [(d, exts)
    for d in vcat(10, 1000:1000:12000)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, nonparametricdistr_insts(
    VecSpecExt[VecPower12(1.5), VecNegEntropy()]
    ))
insts["ext"] = (nothing, nonparametricdistr_insts(
    VecSpecExt[VecPower12EF(1.5), VecNegEntropyEF()]
    ))
return (NonparametricDistrJuMP, insts)
