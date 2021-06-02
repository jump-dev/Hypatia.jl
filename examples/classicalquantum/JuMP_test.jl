
insts = OrderedDict()
insts["minimal"] = [
    ((3, false, false),),
    ((3, false, true),),
    ((3, true, false),),
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
