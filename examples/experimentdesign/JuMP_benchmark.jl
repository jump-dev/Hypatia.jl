
experimentdesign_insts(ext::MatSpecExt) = [
    [(d, ext)
    for d in vcat(3, 10:10:40, 50:50:400)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, experimentdesign_insts.(
    [MatNegLog(), MatNegEntropy(), MatPower12(1.5)]
    ))
insts["ext"] = (nothing, experimentdesign_insts.(
    [MatNegLogEigOrd(), MatNegEntropyEigOrd(), MatPower12EigOrd(1.5)]
    ))
return (ExperimentDesignJuMP, insts)
