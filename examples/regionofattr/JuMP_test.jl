
insts = Dict()
insts["minimal"] = [
    ((4, true),),
    ((4, false), :SOCExpPSD),
    ]
insts["fast"] = [
    ((6, true),),
    ((6, false), :SOCExpPSD),
    ((8, true),),
    ]
insts["slow"] = [
    ((8, false), :SOCExpPSD),
    ((10, true),),
    ((10, false), :SOCExpPSD),
    ]
insts["various"] = vcat(insts["fast"], insts["slow"])
return (RegionOfAttrJuMP, insts)
