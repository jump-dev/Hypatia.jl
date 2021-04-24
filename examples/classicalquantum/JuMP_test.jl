
insts = Dict()
insts["minimal"] = [
    ((1,),),
    ]
insts["fast"] = [
    ((10,),),
    ]
insts["slow"] = Tuple[]
insts["various"] = [
    ((25,),),
    ((50,),),
    ((100,),),
    ]
return (ClassicalQuantum, insts)
