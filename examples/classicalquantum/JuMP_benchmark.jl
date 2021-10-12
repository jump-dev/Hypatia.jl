
classicalquantum_insts(complex::Bool, use_EF::Bool) = [
    [(d, complex, use_EF)
    for d in vcat(3, 25:25:100, 150:50:500)] # includes compile run
    ]

insts = OrderedDict()
insts["compnat"] = (nothing, classicalquantum_insts(true, false))
insts["realnat"] = (nothing, classicalquantum_insts(false, false))
insts["realext"] = (nothing, classicalquantum_insts(false, true))
return (ClassicalQuantum, insts)
