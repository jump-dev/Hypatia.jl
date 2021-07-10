
nonparametricdistr_ds = vcat(10, 2000:2000:18000) # includes compile run
insts = OrderedDict()
insts["nat"] = (nothing, [[(d, VecPower12(1.5)) for d in nonparametricdistr_ds]])
insts["ext"] = (nothing, [[(d, VecPower12EF(1.5)) for d in nonparametricdistr_ds]])
return (NonparametricDistrJuMP, insts)
