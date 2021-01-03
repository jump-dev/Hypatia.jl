
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
return (LotkaVolterraJuMP, insts)
