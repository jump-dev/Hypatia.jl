
insts = Dict()
insts["minimal"] = [
    ((2, 2, true),),
    ((2, 2, false),),
    ]
insts["fast"] = [
    ((2, 3, true),),
    ((2, 3, false),),
    ((5, 6, true),),
    ((5, 6, false),),
    ((10, 20, true),),
    ((10, 20, false),),
    ((20, 40, true),),
    ((20, 40, false),),
    ]
insts["slow"] = [
    ((60, 80, true),),
    ((60, 80, false),),
    ]
return (MatrixQuadraticJuMP, insts)
