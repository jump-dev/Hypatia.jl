
log_ps = vcat(3, 20:20:80, 100:100:1000)
pow_ps = vcat(3, 20:20:80, 100:100:800)
experimentdesign_insts(ext::Union{MatNegLog, MatNegLogDirect, MatNegLogEigOrd}) = [
    [(p, ext) for p in log_ps] # includes compile run
    ]
experimentdesign_insts(ext::Union{MatPower12, MatPower12EigOrd}) = [
    [(p, ext) for p in pow_ps] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, vcat(experimentdesign_insts(MatNegLog()),
    experimentdesign_insts(MatPower12(1.5))))
insts["extdirect"] = (nothing, experimentdesign_insts(MatNegLogDirect()))
insts["extord"] = (nothing, vcat(experimentdesign_insts(MatNegLogEigOrd()),
    experimentdesign_insts(MatPower12EigOrd(1.5))))
insts["logdet"] = (nothing, [[(p, nothing) for p log_ps]])
return (ExperimentDesignJuMP, insts)
