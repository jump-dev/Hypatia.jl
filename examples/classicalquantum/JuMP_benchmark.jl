
classicalquantum_insts(use_EF::Bool) = [
    [(d, false, use_EF)
    for d in vcat(3, 20:20:100, 150:50:450)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, classicalquantum_insts(false))
insts["ext"] = (nothing, classicalquantum_insts(true))
return (ClassicalQuantum, insts)
