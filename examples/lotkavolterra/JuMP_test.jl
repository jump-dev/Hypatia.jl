
insts = Dict()
insts["minimal"] = [
    ((2,),),
    ]
insts["fast"] = [
    ((4,),),
    ]
insts["slow"] = [
    ((6,),),
    ]
return (LotkaVolterraJuMP, insts)
