
insts = Dict()
insts["minimal"] = [
    ((3, true, false, true),),
    ((3, false, true, true),),
    ((3, false, true, false),),
    ((3, true, true, true),),
    ]
insts["fast"] = [
    ((10, true, false, true),),
    ((10, false, true, true),),
    ((10, false, true, false),),
    ((10, true, true, true),),
    ((50, true, false, true),),
    ((50, false, true, true),),
    ((50, false, true, false),),
    ((50, true, true, true),),
    ((200, true, false, true),),
    ((200, false, true, true),),
    ((200, false, true, false),),
    ((200, true, true, true),),
    ]
insts["slow"] = [
    ((3000, true, false, true),),
    ((3000, false, true, true),),
    ((3000, false, true, false),),
    ((3000, true, true, true),),
    ]
return (PortfolioNative, insts)
