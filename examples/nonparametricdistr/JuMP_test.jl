
insts = OrderedDict()
insts["minimal"] = [
    ((2, 2, false),),
    ((4, 1, false),),
    ((4, 2, true),),
    ((4, 3, false),),
    ((4, 3, true),),
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
