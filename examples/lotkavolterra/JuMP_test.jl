
insts = Dict()
insts["minimal"] = [
    ((3,), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((4,), StandardConeOptimizer),
    ]
insts["slow"] = [
    ((6,), StandardConeOptimizer),
    ]
return (LotkaVolterraJuMP, insts)
