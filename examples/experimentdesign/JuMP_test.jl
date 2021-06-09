
insts = OrderedDict()
insts["minimal"] = [
    # rootdet
    ((6, MatNegGeom()),),
    ((6, MatNegGeomEFExp()),),
    ((6, MatNegGeomEFPow()),),
    # tr inv
    ((4, MatInv()),),
    ((4, MatInvEigOrd()),),
    ((4, MatInvDirect()),),
    # tr neglog
    ((4, MatNegLog()),),
    ((4, MatNegLogEigOrd()),),
    ((4, MatNegLogDirect()),),
    # tr negentropy
    ((4, MatNegEntropy()),),
    ((4, MatNegEntropyEigOrd()),),
    # tr power12
    ((5, MatPower12(1.5)),),
    ((5, MatPower12EigOrd(1.5)),),
    # tr neg2sqrt
    ((4, MatNeg2Sqrt()),),
    ((4, MatNeg2SqrtEigOrd()),),
    ]
insts["fast"] = [
    ((100, MatNegGeom()),),
    ((25, MatNegGeomEFExp()),),
    ((25, MatNegGeomEFPow()),),
    ((80, MatInv()),),
    ((12, MatInvEigOrd()),),
    ((50, MatInvDirect()),),
    ((150, MatNegLog()),),
    ((14, MatNegLogEigOrd()),),
    ((30, MatNegLogDirect()),),
    ((120, MatNegEntropy()),),
    ((15, MatNegEntropyEigOrd()),),
    ((100, MatPower12(1.5)),),
    ((10, MatPower12EigOrd(1.5)),),
    ]
insts["various"] = [
    ((800, MatNegGeom()),),
    ((140, MatNegGeomEFExp()),),
    ((80, MatNegGeomEFPow()), nothing, (default_tol_relax = 100,)),
    ((400, MatInv()),),
    ((20, MatInvEigOrd()),),
    ((250, MatInvDirect()),),
    ((300, MatNegLog()),),
    ((24, MatNegLogEigOrd()),),
    ((150, MatNegLogDirect()),),
    ((300, MatNegEntropy()),),
    ((20, MatNegEntropyEigOrd()),),
    ((30, MatPower12(1.5)),),
    ((18, MatPower12EigOrd(1.5)), nothing, (default_tol_relax = 100,)),
    ]
return (ExperimentDesignJuMP, insts)
