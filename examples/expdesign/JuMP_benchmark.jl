
expdesign_insts = [
    [(q, 2q, 2q, 5, use_logdet, !use_logdet, false) for q in vcat(3, 20:20:max_q)] # includes compile run
    for (use_logdet, max_q) in ((false, 300), (true, 240))
    ]

insts = Dict()
insts["nat"] = (nothing, expdesign_insts)
insts["ext"] = (ExpPSDOptimizer, expdesign_insts)
return (ExpDesignJuMP, insts)
