
insts = Dict()
insts["minimal"] = [
    ((4, true),),
    ((4, false),),
    ]
insts["fast"] = [
    ((6, true),),
    ((6, false),),
    ((8, true),),
    ]
insts["slow"] = [
    ((8, false),),
    ]
return (RegionOfAttrJuMP, insts)
