
classicalquantum_ns = vcat(3, 10:10:50, 100:100:500) # includes compile run
insts = OrderedDict()
insts["nat"] = (nothing, [[(n, false, false) for n in classicalquantum_ns]])
insts["ext"] = (nothing, [[(n, false, true) for n in classicalquantum_ns]])
return (ClassicalQuantum, insts)
