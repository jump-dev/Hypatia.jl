
expdesign_instances = [
    [(q, 2q, 2q, 5, use_logdet, !use_logdet, false) for q in vcat(3, 20:20:220)] # includes compile run
    for use_logdet in (false, true)
    ]

instances[ExpDesignJuMP]["nat"] = (nothing, expdesign_instances)
instances[ExpDesignJuMP]["ext"] = (StandardConeOptimizer, expdesign_instances)
