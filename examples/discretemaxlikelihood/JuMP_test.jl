
insts = OrderedDict()
insts["minimal"] = [
    ((2, false),),
    ((2, true),),
    ((4, false),),
    ((4, true),),
    ]
insts["fast"] = [
    ((50, false),),
    ((50, true),),
    ((1000, false),),
    ((250, true),),
    ]
insts["various"] = [
    ((1000, false),),
    ((500, true),),
    ((4000, false),),
    ((1000, true),),
    ((8000, false),),
    ((2000, true),),
    ((10000, false),),
    ]
return (DiscreteMaxLikelihood, insts)
