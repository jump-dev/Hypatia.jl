
insts = Dict()
insts["minimal"] = [
    ((3, true, false),),
    ((3, false, true),),
    ((3, false, true), SOCExpPSDOptimizer),
    ((3, true, true),),
    ]
insts["fast"] = [
    ((10, true, false),),
    ((10, false, true),),
    ((10, false, true), SOCExpPSDOptimizer),
    ((10, true, true),),
    ((50, true, false),),
    ((50, false, true),),
    ((50, false, true), SOCExpPSDOptimizer),
    ((50, true, true),),
    ((400, true, false),),
    ((400, false, true),),
    ((400, false, true), SOCExpPSDOptimizer),
    ((400, true, true),),
    ]
insts["slow"] = [
    ((1000, true, false),),
    ((1000, false, true),),
    ((1000, false, true), SOCExpPSDOptimizer),
    ((1000, true, true),),
    ((2000, true, false),),
    ((2000, false, true), SOCExpPSDOptimizer),
    ((2000, true, true),),
    ((3000, false, true),),
    ]
insts["various"] = [
    ((50, true, false),),
    ((50, false, true),),
    ((50, false, true), SOCExpPSDOptimizer),
    ((50, true, true),),
    ((500, true, false),),
    ((500, false, true),),
    ((500, false, true), SOCExpPSDOptimizer),
    ((500, true, true),),
    ((4000, true, false),),
    ((4000, false, true),),
    ((4000, false, true), SOCExpPSDOptimizer),
    ((4000, true, true),),
    ]
return (PortfolioJuMP, insts)
