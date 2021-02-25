
insts = Dict()
insts["minimal"] = [
    ((2, false, true, true),),
    ((2, false, false, true),),
    ((2, true, true, true),),
    ((2, true, false, true),),
    ((2, false, true, false),),
    ((2, false, false, false),),
    ((2, true, true, false),),
    ((2, true, false, false),),
    ]
insts["fast"] = [
    ((5, false, true, true),),
    ((5, false, false, true),),
    ((5, true, true, true),),
    ((5, true, false, true),),
    ((5, false, true, false),),
    ((5, false, false, false),),
    ((5, true, true, false),),
    ((5, true, false, false),),
    ((20, false, true, true),),
    ((20, false, false, true),),
    ((20, true, true, true),),
    ((20, true, false, true),),
    ((20, false, true, false),),
    ((20, false, false, false),),
    ((20, true, true, false),),
    ((20, true, false, false),),
    ((100, false, true, false),),
    ((100, false, false, false),),
    ]
insts["slow"] = [
    ((100, false, true, true),),
    ((100, false, false, true),),
    ((100, true, true, true),),
    ((100, true, false, true),),
    ((100, true, true, false),),
    ((100, true, false, false),),
    ]
insts["various"] = vcat(insts["fast"], insts["slow"])
return (NearestPSDJuMP, insts)
