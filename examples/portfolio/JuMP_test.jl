
insts = OrderedDict()
insts["minimal"] = [
    ((3, true, false),),
    ((3, false, true),),
    ((3, false, true), :SOCExpPSD),
    ]
insts["fast"] = [
    ((10, true, false),),
    ((10, false, true),),
    ((10, false, true), :SOCExpPSD),
    ((50, true, false),),
    ((50, false, true),),
    ((50, false, true), :SOCExpPSD),
    ((400, true, false),),
    ((400, false, true),),
    ((400, true, false),),
    ((400, false, true),),
    ((400, false, true), :SOCExpPSD),
    ]
insts["various"] = [
    ((50, false, true), :SOCExpPSD),
    ((1000, true, false),),
    ((1000, false, true),),
    ((2000, true, false),),
    ((2000, false, true),),
    ((4000, true, false),),
    ((4000, false, true),),
    ((8000, true, false),),
    ((8000, false, true),),
    ]
return (PortfolioJuMP, insts)
