
insts = Dict()
insts["minimal"] = [
    ((2,), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    ((3,), SOCExpPSDOptimizer),
    ]
insts["slow"] = [
    ((4,), SOCExpPSDOptimizer),
    ]
insts["various"] = vcat(insts["fast"], insts["slow"])
return (LotkaVolterraJuMP, insts)
