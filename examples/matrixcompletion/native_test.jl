
insts = OrderedDict()
insts["minimal"] = [
    ((3, 4, true, true, true, true),),
    ((3, 4, false, true, true, true),),
    ((3, 4, true, false, true, true),),
    ((3, 4, false, false, true, true),),
    ((3, 4, true, true, false, true),),
    ((3, 4, false, false, false, true),),
    ((3, 4, true, true, false, false),),
    ((3, 4, false, false, false, false),),
    ]
insts["fast"] = [
    ((12, 24, true, true, true, true),),
    ((12, 24, false, true, true, true),),
    ((12, 24, true, false, true, true),),
    ((12, 24, false, false, true, true),),
    ((12, 24, true, true, false, true),),
    ((12, 24, false, false, false, true),),
    ((12, 24, true, true, false, false),),
    ((12, 24, false, false, false, false),),
    ]
insts["various"] = [
    ((5, 100, true, true, true, true),),
    ((5, 100, true, true, false, true),),
    ((10, 200, true, true, true, true),),
    ((10, 200, true, true, false, true),),
    ]
return (MatrixCompletionNative, insts)
