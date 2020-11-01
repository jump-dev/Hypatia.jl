
insts = Dict()
insts["minimal"] = [
    ((3, true, false),),
    ((3, false, true),),
    ((3, false, true), StandardConeOptimizer),
    ((3, true, true),),
    ]
insts["fast"] = [
    ((10, true, false),),
    ((10, false, true),),
    ((10, false, true), StandardConeOptimizer),
    ((10, true, true),),
    ((50, true, false),),
    ((50, false, true),),
    ((50, false, true), StandardConeOptimizer),
    ((50, true, true),),
    ((400, true, false),),
    ((400, false, true),),
    ((400, true, true),),
    ((400, true, false),),
    ((400, false, true),),
    ((400, false, true), StandardConeOptimizer),
    ((400, true, true),),
    ]
insts["slow"] = [
    ((1000, true, false),),
    ((1000, false, true),),
    ((1000, false, true), StandardConeOptimizer),
    ((1000, true, true),),
    ((2000, true, false),),
    ((2000, false, true), StandardConeOptimizer),
    ((2000, true, true),),
    ((3000, false, true),),
    ]
return (PortfolioJuMP, insts)
