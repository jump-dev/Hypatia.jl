
experimentdesign_insts(ext::MatSpecExt) = [
    [(d, ext)
    for d in vcat(3, 25:25:100, 200:100:400)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(
    experimentdesign_insts(MatNegSqrt()),
    experimentdesign_insts(MatNegSqrtConj()),
    experimentdesign_insts(MatNegPower01(1/3)),
    experimentdesign_insts(MatNegPower01Conj(1/3)),
    experimentdesign_insts(MatPower12(1.5)),
    ))
insts["ext"] = (nothing, vcat(
    experimentdesign_insts(MatNegSqrtEigOrd()),
    experimentdesign_insts(MatNegSqrtConjDirect()),
    experimentdesign_insts(MatNegPower01EigOrd(1/3)),
    experimentdesign_insts(MatNegPower01ConjEigOrd(1/3)),
    experimentdesign_insts(MatPower12EigOrd(1.5)),
    ))
return (ExperimentDesignJuMP, insts)
