
classicalquantum_insts(complex::Bool, use_EF::Bool) = [
    [(d, complex, use_EF)
    for d in vcat(3, 10:10:40, 50:25:100, 150:50:700)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, classicalquantum_insts(true, false))
insts["ext"] = (nothing, classicalquantum_insts(true, true))
return (ClassicalQuantum, insts)
