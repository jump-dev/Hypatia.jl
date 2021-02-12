
insts = Dict()
insts["minimal"] = [
    ((3, 2, true, 0),),
    ((3, 2, false, 0),),
    ((3, 2, true, 10),),
    ((3, 2, false, 10),),
    ]
insts["fast"] = [
    ((5, 3, true, 0),),
    ((5, 3, false, 0),),
    ((5, 3, true, 10),),
    ((5, 3, false, 10),),
    ((30, 10, true, 0),),
    ((30, 10, false, 0), (default_tol_relax = 1000,)),
    ((30, 10, true, 10),),
    ((30, 10, false, 10),),
    ]
insts["slow"] = Tuple[]
insts["various"] = [
    ((5, 3, true, 10),),
    ((5, 3, false, 10),),
    ((10, 6, true, 10),),
    ((10, 6, false, 10),),
    ((20, 12, true, 10),),
    ((20, 12, false, 10),),
    ]
return (SparsePCANative, insts)
