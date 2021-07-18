
experimentdesign_insts(ext::MatSpecExt) = [
    [(d, ext)
    for d in vcat(3, 25:25:100, 200:100:400)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(
    experimentdesign_insts(MatNegLog()),
    experimentdesign_insts(MatNegEntropy()),
    experimentdesign_insts(MatPower12(1.5)),
    ))
insts["ext"] = (nothing, vcat(
    experimentdesign_insts(MatNegLogEigOrd()),
    experimentdesign_insts(MatNegEntropyEigOrd()),
    experimentdesign_insts(MatPower12EigOrd(1.5)),
    ))
return (ExperimentDesignJuMP, insts)
