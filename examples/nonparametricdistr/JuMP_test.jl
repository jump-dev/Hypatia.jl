
relaxed_tols = (default_tol_relax = 100,)
insts = OrderedDict()
insts["minimal"] = [
    ((2, false),),
    ((2, true),),
    ((4, false),),
    ((4, true), nothing, relaxed_tols),
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
