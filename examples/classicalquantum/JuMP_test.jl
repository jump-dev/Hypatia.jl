
insts = OrderedDict()
insts["minimal"] = [
    ((2, false),),
    ((2, true),),
    ]
insts["fast"] = [
    ((10, false),),
    ((10, true),),
    ]
insts["various"] = [
    ((25, false),),
    ((25, true),),
    ((50, false),),
    ((50, true),),
    ((100, false),),
    ((100, true),),
    ]
return (ClassicalQuantum, insts)
