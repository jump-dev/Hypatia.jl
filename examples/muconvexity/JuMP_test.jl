
insts = Dict()
insts["minimal"] = [
    ((:poly1, :dom1, true, -4),),
    ((:poly1, :dom1, false, -4), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    ((:poly1, :dom2, true, -4),),
    ((:poly1, :dom1, false, -4), SOCExpPSDOptimizer),
    ((:poly1, :dom2, false, -4), SOCExpPSDOptimizer),
    ((:poly2, :dom3, true, -2),),
    ((:poly2, :dom4, true, -2),),
    ((:poly2, :dom3, false, -2), SOCExpPSDOptimizer),
    ((:poly2, :dom4, false, -2), SOCExpPSDOptimizer),
    ]
insts["slow"] = Tuple[]
insts["various"] = insts["fast"]
return (MuConvexityJuMP, insts)
