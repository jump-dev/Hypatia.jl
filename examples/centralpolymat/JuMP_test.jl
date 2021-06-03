
insts = OrderedDict()
insts["minimal"] = [
    # rootdet
    ((1, 2, MatNegGeom()),),
    ((1, 2, MatNegGeomEFExp()),),
    ((1, 2, MatNegGeomEFPow()),),
    # tr inv
    ((1, 2, MatInv()),),
    ((1, 2, MatInvEigOrd()),),
    ((1, 2, MatInvDirect()),),
    ]
insts["fast"] = [
    ((1, 10, true),),
    ((1, 10, true),),
    ((1, 15, false),),
    ((1, 15, false),),
    ((2, 3, true),),
    ((2, 3, true),),
    ((2, 3, false),),
    ((2, 3, false),),
    ((2, 6, true),),
    ((2, 5, true),),
    ((2, 7, false),),
    ((2, 6, false),),
    ((3, 2, true),),
    ((3, 2, false),),
    ((3, 4, true),),
    ((3, 4, false),),
    ((7, 2, true),),
    ((7, 2, true),),
    ((7, 2, false),),
    ((7, 2, false),),
    ]
insts["various"] = [
    ((2, 5, true),),
    ((2, 5, true),),
    ((2, 5, false),),
    ((2, 5, false),),
    ((2, 5, false),),
    ((2, 10, true),),
    ((2, 10, true),),
    ((2, 10, false),),
    ((2, 10, false),),
    ((2, 10, false),),
    ((6, 3, true),),
    ((6, 3, false),),
    ((8, 3, true),),
    ((8, 3, false),),
    ]
return (CentralPolyMatJuMP, insts)
