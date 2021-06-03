
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
    ((1, 10, MatNegGeomEFExp()),),
    ((1, 15, MatNegGeom()),),
    ((2, 3, MatNegGeomEFPow()),),
    ((2, 3, MatInvEigOrd()),),
    ((2, 5, MatInvDirect()),),
    ((2, 6, MatInv()),),
    ((2, 6, MatNegEntropy()),),
    ((2, 7, MatNegLog()),),
    ((3, 2, MatNegEntropyEigOrd()),),
    ((3, 2, MatPower12EigOrd(1.5)),),
    ((3, 4, MatPower12(1.5)),),
    ((3, 4, MatNegGeom()),),
    ((7, 2, MatNegLog()),),
    ((7, 2, MatInv()),),
    ((7, 2, MatNegEntropy()),),
    ((7, 2, MatPower12(1.5)),),
    ]
insts["various"] = [
    ((2, 5, MatInvEigOrd()),),
    ((2, 5, MatNegLogEigOrd()),),
    ((2, 4, MatPower12EigOrd(1.5)),),
    ((2, 4, MatNegEntropyEigOrd()),),
    ((2, 10, MatNegGeomEFExp()),),
    ((2, 8, MatNegGeomEFPow()),),
    ((2, 14, MatNegGeom()),),
    ((2, 12, MatInv()),),
    ((2, 10, MatNegLog()),),
    ((2, 8, MatNegEntropy()),),
    ((2, 6, MatPower12(1.5)),),
    ((3, 6, MatInvDirect()),),
    ((3, 5, MatNegLogDirect()),),
    ((3, 6, MatNegEntropy()),),
    ((3, 7, MatNegLog()),),
    ((3, 5, MatPower12(1.5)),),
    ((5, 3, MatInvDirect()),),
    ((5, 3, MatNegLogDirect()),),
    ((8, 3, MatNegGeom()),),
    ((8, 3, MatInv()),),
    ]
return (CentralPolyMatJuMP, insts)
