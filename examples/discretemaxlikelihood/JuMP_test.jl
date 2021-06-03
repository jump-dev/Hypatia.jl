
insts = OrderedDict()
insts["minimal"] = [
    ((2, false),),
    ((2, true),),
    ((4, false),),
    ((4, true),),
    ]
insts["fast"] = [
    ((10,),),
    ((100,),),
    ((1000,),),
    ]
insts["various"] = [
    ((2500,),),
    ((5000,),),
    ((10000,),),
    ]
return (DiscreteMaxLikelihood, insts)
