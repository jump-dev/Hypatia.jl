
insts = OrderedDict()
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
insts["various"] = [
    ((15, 20, true),),
    ((15, 20, false),),
    ((30, 40, true),),
    ((30, 40, false),),
    ((60, 80, true),),
    ((60, 80, false),),
    ((30, 160, true),),
    ((30, 160, false),),
    ]
return (MatrixQuadraticJuMP, insts)
