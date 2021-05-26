
insts = OrderedDict()
insts["minimal"] = [
    ((2,),),
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
