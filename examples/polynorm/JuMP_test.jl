
insts = Dict()
insts["minimal"] = [
    ((1, 2, 2),),
    ]
insts["fast"] = [
    ((2, 2, 2),),
    ((2, 1, 3),),
    ]
insts["slow"] = Tuple[]
return (PolyNormJuMP, insts)
