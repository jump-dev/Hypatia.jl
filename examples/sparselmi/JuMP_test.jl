
insts = Dict()
insts["minimal"] = [
    ((1, 3, 2, false, true, false, false, false),),
    ((1, 3, 2, false, false, true, false, false),),
    ((1, 3, 2, true, false, false, true, false),),
    ((1, 3, 2, true, false, true, false, false),),
    ((1, 3, 2, true, false, false, true, true),),
    ]
# TODO
insts["fast"] = []
insts["slow"] = []
insts["various"] = insts["minimal"]
return (SparseLMIJuMP, insts)