
insts = OrderedDict()
insts["minimal"] = [
    ((4, true),),
    ((4, false), :SOCExpPSD),
    ]
insts["fast"] = [
    ((6, true),),
    ((6, false), :SOCExpPSD),
    ((8, true),),
    ]
insts["various"] = vcat(insts["fast"], [
    ((8, false), :SOCExpPSD),
    ((10, true),),
    ((10, false), :SOCExpPSD),
    ])
return (RegionOfAttrJuMP, insts)
