
classicalquantum_insts(complex::Bool, use_EF::Bool) = [
    [(d, complex, use_EF)
    for d in vcat(3, 25:25:100, 150:50:300)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, classicalquantum_insts(true, false))
insts["ext"] = (nothing, classicalquantum_insts(true, true))
return (ClassicalQuantum, insts)
