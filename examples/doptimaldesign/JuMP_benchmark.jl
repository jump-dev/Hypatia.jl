
doptimaldesign_insts = [
    [(q, 2q, 2q, 5, use_logdet, !use_logdet, false) for q in vcat(3, 25:25:300)] # includes compile run
    for use_logdet in (false, true)
    ]

insts = Dict()
insts["nat"] = (nothing, doptimaldesign_insts)
insts["ext"] = (SOCExpPSDOptimizer, doptimaldesign_insts)
return (DOptimalDesignJuMP, insts)
