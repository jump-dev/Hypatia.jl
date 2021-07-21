
insts = OrderedDict()
insts["minimal"] = [
    # tr neglog
    ((3, MatLogdetCone()),),
    ((3, MatNegLog()),),
    ((3, MatNegLogEigOrd()),),
    ((3, MatNegLogDirect()),),
    # tr negentropy
    ((3, MatNegEntropy()),),
    ((3, MatNegEntropyEigOrd()),),
    # tr negsqrt
    ((3, MatNegSqrt()),),
    ((3, MatNegSqrtEigOrd()),),
    # tr negpower01
    ((3, MatNegPower01(0.7)),),
    ((3, MatNegPower01(0.7)),),
    # tr power12
    ((3, MatPower12(1.3)),),
    ((3, MatPower12EigOrd(1.3)),),
    ]
insts["fast"] = [
    ((40, MatNegGeom()),),
    ((15, MatNegGeomEFExp()),),
    ((15, MatNegGeomEFPow()),),
    ((40, MatNegSqrtConj()),),
    ((8, MatNegSqrtConjEigOrd()),),
    ((20, MatNegSqrtConjDirect()),),
    ((50, MatLogdetCone()),),
    ((30, MatNegLog()),),
    ((6, MatNegLogEigOrd()),),
    ((15, MatNegLogDirect()),),
    ((30, MatNegEntropy()),),
    ((8, MatNegEntropyEigOrd()),),
    ((30, MatPower12(1.5)),),
    ((7, MatPower12EigOrd(1.5)),),
    ]
insts["various"] = [
    ((100, MatNegGeom()),),
    ((50, MatNegGeomEFExp()),),
    ((30, MatNegGeomEFPow()),),
    ((100, MatNegSqrtConj()),),
    ((12, MatNegSqrtConjEigOrd()),),
    ((50, MatNegSqrtConjDirect()),),
    ((150, MatLogdetCone()),),
    ((100, MatNegLog()),),
    ((18, MatNegLogEigOrd()), nothing, (default_tol_relax = 1000,)),
    ((80, MatNegLogDirect()),),
    ((75, MatNegEntropy()),),
    ((14, MatNegEntropyEigOrd()),),
    ((30, MatPower12(1.5)),),
    ((6, MatPower12EigOrd(1.5)),),
    ]
return (CovarianceEstJuMP, insts)
