
insts = Dict()
insts["minimal"] = [
    ((1, 2, 2, 2),),
    ]
insts["fast"] = [
    ((2, 2, 3, 2),),
    ((3, 3, 3, 3),),
    ((3, 3, 5, 4),),
    ((5, 2, 5, 3),),
    ((1, 30, 2, 30),),
    ((10, 1, 3, 1),),
    ]
insts["slow"] = [
    ((4, 5, 4, 6),),
    ((2, 30, 4, 30),),
    ]
insts["various"] = insts["fast"]
return (EnvelopeJuMP, insts)
