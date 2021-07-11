
classicalquantum_ms = vcat(3, 10:10:50, 100:100:500) # includes compile run
insts = OrderedDict()
insts["nat"] = (nothing, [[(m, false, false) for m in classicalquantum_ms]])
insts["ext"] = (nothing, [[(m, false, true) for m in classicalquantum_ms]])
return (ClassicalQuantum, insts)
