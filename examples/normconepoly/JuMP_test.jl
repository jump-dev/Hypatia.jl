
insts = OrderedDict()
insts["minimal"] = [
    ((:polys1, 2, true, true),),
    ((:polys1, 2, true, false),),
    ]
insts["fast"] = [
    ((:polys2, 2, true, true),),
    ((:polys3, 2, true, true),),
    ((:polys4, 4, true, true),),
    ((:polys5, 2, false, true),),
    ((:polys6, 2, false, true),),
    ((:polys7, 2, false, true),),
    ((:polys8, 2, false, true),),
    ((:polys9, 2, false, true),),
    ((:polys2, 2, true, false),),
    ((:polys3, 2, true, false),),
    ((:polys4, 4, true, false),),
    ((:polys5, 2, false, false),),
    ((:polys6, 2, false, false),),
    ((:polys7, 2, false, false),),
    ((:polys8, 2, false, false),),
    ((:polys9, 2, false, false),),
    ]
insts["various"] = [
    ((:polys2, 2, true, true),),
    ((:polys3, 2, true, true),),
    ((:polys4, 4, true, true),),
    ((:polys7, 2, false, true),),
    ((:polys8, 2, false, true),),
    ((:polys2, 2, true, false),),
    ((:polys3, 2, true, false),),
    ((:polys4, 4, true, false),),
    ((:polys7, 2, false, false),),
    ((:polys8, 2, false, false),),
    ]
return (NormConePoly, insts)
