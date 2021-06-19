
insts = OrderedDict()
insts["minimal"] = [
    # tr neglog
    ((3, MatNegLog()),),
    ((3, MatNegLogEigOrd()),),
    ((3, MatNegLogDirect()),),
    # tr negentropy
    ((3, MatNegEntropy()),),
    ((3, MatNegEntropyEigOrd()),),
    # tr power12
    ((3, MatPower12(1.5)),),
    ((3, MatPower12EigOrd(1.5)),),
    # tr neg2sqrt
    ((3, MatNeg2Sqrt()),),
    ((3, MatNeg2SqrtEigOrd()),),
    ]
insts["fast"] = [
    ((40, MatNegGeom()),),
    ((15, MatNegGeomEFExp()),),
    ((15, MatNegGeomEFPow()),),
    ((40, MatInv()),),
    ((8, MatInvEigOrd()),),
    ((20, MatInvDirect()),),
    ((30, MatNegLog()),),
    ((12, MatNegLogEigOrd()),),
    ((15, MatNegLogDirect()),),
    ((30, MatNegEntropy()),),
    ((8, MatNegEntropyEigOrd()),),
    ((30, MatPower12(1.5)),),
    ((10, MatPower12EigOrd(1.5)),),
    ]
insts["various"] = [
    ((100, MatNegGeom()),),
    ((50, MatNegGeomEFExp()),),
    ((30, MatNegGeomEFPow()),),
    ((100, MatInv()),),
    ((12, MatInvEigOrd()),),
    ((50, MatInvDirect()),),
    ((100, MatNegLog()),),
    ((18, MatNegLogEigOrd()), nothing, (default_tol_relax = 1000,)),
    ((80, MatNegLogDirect()),),
    ((75, MatNegEntropy()),),
    ((14, MatNegEntropyEigOrd()),),
    ((30, MatPower12(1.5)),),
    ((6, MatPower12EigOrd(1.5)),),
    ]
insts["natvext"] = [
    # tr neglog
    # ((600, MatNegLog()),), # out of memory in densify
    ((150, MatNegLog()),),
    # ((200, MatNegLog()),), # far from converging after 2628.812 seconds
    # ((100, MatNegLogDirect()),), # 17 iterations and 373.72 seconds
    ((120, MatNegLogDirect()),),
    # ((20, MatNegLogEigOrd()), nothing, (default_tol_relax = 1000,)), # 36 iterations and 63.414 seconds
    ((40, MatNegLogEigOrd()),),
    # tr negentropy
    ((150, MatNegEntropy()),),
    ((40, MatNegEntropyEigOrd()),),
    # tr power12
    ((150, MatPower12(1.5)),),
    ((40, MatPower12EigOrd(1.5)),),
    ]
return (CovarianceEstJuMP, insts)
