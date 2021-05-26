
insts = OrderedDict()
insts["minimal"] = [
    ((1, 3, 2, false, true, false, false, false),),
    ((1, 3, 2, false, false, true, false, false),),
    ((1, 3, 2, true, false, false, true, false),),
    ((1, 3, 2, true, false, true, false, false),),
    ((1, 3, 2, true, false, false, true, true),),
    ]
# insts["fast"] = [] # TODO
insts["various"] = [
    ((5, 20, 20, false, true, false, false, false),),
    ((5, 20, 20, false, false, true, false, false),),
    ((5, 20, 20, true, false, false, true, false),),
    ((5, 20, 20, true, false, true, false, false),),
    ((5, 20, 20, true, false, false, true, true),),
    ((10, 30, 20, false, true, false, false, false),),
    ((10, 30, 20, false, false, true, false, false),),
    ((10, 30, 20, true, false, false, true, false),),
    ((10, 30, 20, true, false, true, false, false),),
    ((10, 30, 20, true, false, false, true, true),),
    ((8, 50, 30, false, true, false, false, false),),
    ((8, 50, 30, false, false, true, false, false),),
    ((8, 50, 30, true, false, false, true, false),),
    ((8, 50, 30, true, false, true, false, false),),
    ((8, 50, 30, true, false, false, true, true),),
    ]
return (SparseLMIJuMP, insts)
