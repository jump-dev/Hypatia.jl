
insts = Dict()
insts["minimal"] = [
    ((:poly1, :dom1, true, -4),),
    ]
insts["fast"] = [
    ((:poly1, :dom2, true, -4),),
    ((:poly1, :dom1, false, -4),),
    ((:poly1, :dom2, false, -4),),
    ((:poly2, :dom3, true, -2),),
    ((:poly2, :dom4, true, -2),),
    ((:poly2, :dom3, false, -2),),
    ((:poly2, :dom4, false, -2),),
    ]
insts["slow"] = Tuple[]
return (MuConvexityJuMP, insts)
