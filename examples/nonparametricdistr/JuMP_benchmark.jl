
nonparametricdistr_insts(exts::Vector{VecSpecExt}) = [
    [(d, exts)
    for d in vcat(10, 1000:1000:8000)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, nonparametricdistr_insts(
    VecSpecExt[VecNegEntropy(), VecNegLog(), VecPower12(1.5)]
    ))
insts["ext"] = (nothing, nonparametricdistr_insts(
    VecSpecExt[VecNegEntropyEF(), VecNegLogEF(), VecPower12EF(1.5)]
    ))
return (NonparametricDistrJuMP, insts)
