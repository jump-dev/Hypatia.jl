
insts = Dict()
insts["minimal"] = [
    ((2,),),
    ((3,),),
    ]
insts["fast"] = [
    ((25,),),
    ((50,),),
    ((100,),),
    ((250,),),
    ]
insts["slow"] = []
insts["various"] = [
    ((25,),),
    ((50,),),
    ((100,),),
    ((250,),),
    ]
return (NonparametricDistrJuMP, insts)
