
insts = OrderedDict()
insts["minimal"] = [
    # rootdet
    ((3, MatNegGeom()),),
    ((3, MatNegGeomEFExp()),),
    ((3, MatNegGeomEFPow()),),
    # tr inv
    ((2, MatInv()),),
    ((2, MatInvEigOrd()),),
    ((2, MatInvDirect()),),
    # tr neglog
    ((2, MatLogdetCone()),),
    ((2, MatNegLog()),),
    ((2, MatNegLogEigOrd()),),
    ((2, MatNegLogDirect()),),
    # tr negentropy
    ((2, MatNegEntropy()),),
    ((2, MatNegEntropyEigOrd()),),
    # tr power12
    ((3, MatPower12(1.5)),),
    ((3, MatPower12EigOrd(1.5)),),
    # tr neg2sqrt
    ((2, MatNeg2Sqrt()),),
    ((2, MatNeg2SqrtEigOrd()),),
    ]
insts["fast"] = [
    ((50, MatNegGeom()),),
    ((12, MatNegGeomEFExp()),),
    ((12, MatNegGeomEFPow()),),
    ((40, MatInv()),),
    ((6, MatInvEigOrd()),),
    ((25, MatInvDirect()),),
    ((100, MatLogdetCone()),),
    ((75, MatNegLog()),),
    ((7, MatNegLogEigOrd()),),
    ((15, MatNegLogDirect()),),
    ((60, MatNegEntropy()),),
    ((8, MatNegEntropyEigOrd()),),
    ((50, MatPower12(1.5)),),
    ((5, MatPower12EigOrd(1.5)),),
    ]
insts["various"] = [
    ((400, MatNegGeom()),),
    ((70, MatNegGeomEFExp()),),
    ((40, MatNegGeomEFPow()), nothing, (default_tol_relax = 100,)),
    ((200, MatInv()),),
    ((10, MatInvEigOrd()),),
    ((125, MatInvDirect()),),
    ((300, MatLogdetCone()),),
    ((150, MatNegLog()),),
    ((12, MatNegLogEigOrd()),),
    ((75, MatNegLogDirect()),),
    ((150, MatNegEntropy()),),
    ((10, MatNegEntropyEigOrd()),),
    ((15, MatPower12(1.5)),),
    ((9, MatPower12EigOrd(1.5)), nothing, (default_tol_relax = 100,)),
    ]
return (ExperimentDesignJuMP, insts)
