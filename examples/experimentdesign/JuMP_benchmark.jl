
experimentdesign_insts(ext::MatSpecExt) = [
    [(d, ext)
    for d in vcat(3, 25:25:100, 150, 200:100:700)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(
    experimentdesign_insts(MatLogdetCone()),
    experimentdesign_insts(MatNegLog()),
    experimentdesign_insts(MatNegSqrt()),
    experimentdesign_insts(MatNegSqrtConj()),
    experimentdesign_insts(MatNegPower01(1/3)),
    ))
insts["ext"] = (nothing, vcat(
    experimentdesign_insts(MatNegLogDirect()),
    experimentdesign_insts(MatNegSqrtEigOrd()),
    experimentdesign_insts(MatNegSqrtConjDirect()),
    experimentdesign_insts(MatNegPower01EigOrd(1/3)),
    ))
return (ExperimentDesignJuMP, insts)
