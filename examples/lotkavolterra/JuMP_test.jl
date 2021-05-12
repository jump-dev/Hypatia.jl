
relaxed_tols = (default_tol_relax = 1000,)
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
insts["various"] = [
    ((3,), SOCExpPSDOptimizer),
    ((4,), SOCExpPSDOptimizer, relaxed_tols),
    ((5,), SOCExpPSDOptimizer, relaxed_tols),
    ]
return (LotkaVolterraJuMP, insts)
