
relaxed_tols = (default_tol_relax = 100,)
insts = OrderedDict()
insts["minimal"] = [
    # rootdet
    ((2, 2, MatNegGeom()),),
    ((2, 2, MatNegGeomEFExp()),),
    ((2, 2, MatNegGeomEFPow()),),
    # tr inv
    ((2, 2, MatInv()),),
    ((2, 2, MatInvEigOrd()),),
    ((2, 2, MatInvDirect()),),
    # tr neg2sqrt
    ((1, 2, MatNeg2Sqrt()),),
    ((1, 2, MatNeg2SqrtEigOrd()),),
    # tr negexp1 (domain not positive)
    ((1, 2, MatNegExp1()),),
    ((1, 2, MatNegExp1EigOrd()),),
    # tr power12conj (domain not positive)
    ((1, 2, MatPower12Conj(1.5)),),
    ((1, 2, MatPower12ConjEigOrd(1.5)),),
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
    ((3, 2, MatNegExp1()),),
    ((3, 2, MatNegExp1EigOrd()),),
    ((3, 2, MatPower12Conj(1.5)),),
    ((3, 2, MatPower12ConjEigOrd(1.5)),),
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
    ((2, 4, MatPower12EigOrd(1.5)), nothing, relaxed_tols),
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
    ((3, 6, MatNegLog()),),
    ((3, 5, MatPower12(1.5)),),
    ((3, 5, MatNegExp1()),),
    ((3, 4, MatPower12Conj(1.5)),),
    ((5, 2, MatNegExp1EigOrd()), nothing, relaxed_tols),
    ((5, 2, MatPower12ConjEigOrd(1.5)),),
    ((5, 3, MatInvDirect()),),
    ((5, 3, MatNegLogDirect()),),
    ((6, 3, MatNegGeom()),),
    ((6, 3, MatInv()),),
    ]
return (CentralPolyMatJuMP, insts)
