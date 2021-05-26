
insts = OrderedDict()
insts["minimal"] = [
    ((3, true, false, true),),
    ((3, false, true, true),),
    ((3, false, true, false),),
    ]
insts["fast"] = [
    ((10, true, false, true),),
    ((10, false, true, true),),
    ((10, false, true, false),),
    ((50, true, false, true),),
    ((50, false, true, true),),
    ((50, false, true, false),),
    ((200, true, false, true),),
    ((200, false, true, true),),
    ((200, false, true, false),),
    ]
return (PortfolioNative, insts)
