
insts = OrderedDict()
insts["minimal"] = [
    ((:poly1, :dom1, true, -4),),
    ((:poly1, :dom1, false, -4), :SOCExpPSD),
    ]
insts["fast"] = [
    ((:poly1, :dom2, true, -4),),
    ((:poly1, :dom1, false, -4), :SOCExpPSD),
    ((:poly1, :dom2, false, -4), :SOCExpPSD),
    ((:poly2, :dom3, true, -2),),
    ((:poly2, :dom4, true, -2),),
    ((:poly2, :dom3, false, -2), :SOCExpPSD),
    ((:poly2, :dom4, false, -2), :SOCExpPSD),
    ]
insts["various"] = insts["fast"]
return (ConvexityParameterJuMP, insts)
