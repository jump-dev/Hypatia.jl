
insts = Dict()
insts["minimal"] = [
    ((3,), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    ((4,), SOCExpPSDOptimizer),
    ]
insts["slow"] = [
    ((6,), SOCExpPSDOptimizer),
    ]
return (LotkaVolterraJuMP, insts)
