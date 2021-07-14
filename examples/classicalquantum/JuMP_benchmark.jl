
classicalquantum_insts(use_EF::Bool) = [
    [(d, false, use_EF)
    for d in vcat(3, 10:10:50, 100:100:500)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, classicalquantum_insts(false))
insts["ext"] = (nothing, classicalquantum_insts(true))
return (ClassicalQuantum, insts)
