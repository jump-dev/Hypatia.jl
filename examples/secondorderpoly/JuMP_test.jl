
insts = Dict()
insts["minimal"] = [
    ((:polys1, 2, true),),
    ]
insts["fast"] = [
    ((:polys2, 2, true),),
    ((:polys3, 2, true),),
    ((:polys4, 4, true),),
    ((:polys5, 2, false),),
    ((:polys6, 2, false),),
    ((:polys7, 2, false),),
    ((:polys8, 2, false),),
    ((:polys9, 2, false),),
    ]
insts["slow"] = Tuple[]
insts["various"] = insts["fast"]
return (SecondOrderPolyJuMP, insts)
