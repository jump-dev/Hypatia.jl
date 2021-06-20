
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
    # ((140, MatNegLog()),), # 43 iterations and 1196.353 seconds
    # ((150, MatNegLog()),), # 48 iterations and 2017.965 seconds
    # ((200, MatNegLog()),), # far from converging after 2628.812 seconds
    # ((600, MatNegLog()),), # out of memory in densify
    # ((100, MatNegLogDirect()),), # 17 iterations and 373.72 seconds
    # ((120, MatNegLogDirect()),), # 18 iterations and 1055.078 seconds
    # ((125, MatNegLogDirect()),), # 17 iterations and 1301.223 seconds
    # ((20, MatNegLogEigOrd()), nothing, (default_tol_relax = 1000,)), # 36 iterations and 63.414 seconds
    # ((25, MatNegLogEigOrd()),), # slow progress, relaxing tols won't help
    # ((30, MatNegLogEigOrd()),), # numerical failure and far from converging after 1972.966 seconds
    # ((35, MatNegLogEigOrd()),), # far from converging after 2531.325 seconds
    # tr negentropy
    # ((150, MatNegEntropy()),), # 33 iterations and 1323.408 seconds
    # ((35, MatNegEntropyEigOrd()),), # far from converging after 2532.218 second
    # ((25, MatNegEntropyEigOrd()),), # slow progress and far from converging after 415.536 seconds
    # tr power12
    # ((150, MatPower12(1.5)),), # 33 iterations and 1311.723 seconds
    # ((155, MatPower12(1.5)),), # 30 iterations and 1418.682 second
    # ((35, MatPower12EigOrd(1.5)),), # tl
    # ((25, MatPower12EigOrd(1.5)),), # slow progress after 605.096 seconds, 1e-5 type viol
    # tr neglog
    ((10, MatNegLogEigOrd()),),
    ((20, MatNegLogEigOrd()),),
    ((30, MatNegLogEigOrd()),),
    ((40, MatNegLogEigOrd()),),
    ((50, MatNegLogEigOrd()),),
    #
    ((10, MatNegLogDirect()),),
    ((20, MatNegLogDirect()),),
    ((30, MatNegLogDirect()),),
    ((40, MatNegLogDirect()),),
    ((50, MatNegLogDirect()),),
    ((75, MatNegLogDirect()),),
    ((100, MatNegLogDirect()),),
    ((125, MatNegLogDirect()),),
    ((150, MatNegLogDirect()),),
    #
    ((10, MatNegLog()),),
    ((20, MatNegLog()),),
    ((30, MatNegLog()),),
    ((40, MatNegLog()),),
    ((50, MatNegLog()),),
    ((75, MatNegLog()),),
    ((100, MatNegLog()),),
    ((125, MatNegLog()),),
    ((150, MatNegLog()),),
    ((175, MatNegLog()),),
    ]
return (CovarianceEstJuMP, insts)
