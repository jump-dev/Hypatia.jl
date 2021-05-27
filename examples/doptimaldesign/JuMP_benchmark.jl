
doptimaldesign_insts = [
    [(q, 2q, 2q, 5, use_logdet, !use_logdet, false)
    for q in vcat(3, 25:25:300)] # includes compile run
    for use_logdet in (false, true)
    ]

insts = OrderedDict()
insts["nat"] = (nothing, doptimaldesign_insts)
insts["ext"] = (:SOCExpPSD, doptimaldesign_insts)
return (DOptimalDesignJuMP, insts)
