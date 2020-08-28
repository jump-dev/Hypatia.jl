
insts = Dict()
insts["minimal"] = [
    ((3, true, false, true, false),),
    ((3, false, true, true, false),),
    ((3, false, true, false, false),),
    ((3, true, true, true, false),),
    ]
insts["fast"] = [
    ((10, true, false, true, false),),
    ((10, false, true, true, false),),
    ((10, false, true, false, false),),
    ((10, true, true, true, false),),
    ((50, true, false, true, false),),
    ((50, false, true, true, false),),
    ((50, false, true, false, false),),
    ((50, true, true, true, false),),
    ((400, true, false, true, false),),
    ((400, false, true, true, false),),
    ((400, false, true, false, false),),
    ((400, true, true, true, false),),
    ]
insts["slow"] = [
    ((3000, true, false, true, false),),
    ((3000, false, true, true, false),),
    ((3000, false, true, false, false),),
    ((3000, true, true, true, false),),
    ]
insts["linops"] = [
    ((20, true, false, true, true),),
    ((20, false, true, true, true),),
    ((20, false, true, false, true),),
    ((20, true, true, true, true),),
    ]
return (PortfolioNative, insts)
