
expdesign_insts = [
    [(q, 2q, 2q, 5, use_logdet, !use_logdet, false) for q in vcat(3, 25:25:300)] # includes compile run
    for use_logdet in (false, true)
    ]

insts = Dict()
insts["nat"] = (nothing, expdesign_insts)
insts["ext"] = (ExpPSDOptimizer, expdesign_insts)
return (ExpDesignJuMP, insts)
