
insts = OrderedDict()
insts["minimal"] = [
    ((2,),),
    ((4,),),
    ]
insts["fast"] = [
    ((25,),),
    ((50,),),
    ((100,),),
    ((250,),),
    ]
insts["various"] = [
    ((25,),),
    ((50,),),
    ((100,),),
    ((250,),),
    ((1000,),),
    ((2500,),),
    ]
return (NonparametricDistrJuMP, insts)
