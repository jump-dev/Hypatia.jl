
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
insts["natvext"] = [
    # rootdet
    # ((1000, MatNegGeom()),),
    # ((150, MatNegGeomEFExp()),),
    # ((300, MatNegGeomEFExp()),),
    # ((450, MatNegGeomEFExp()),),
    # ((100, MatNegGeomEFPow()),),
    # ((200, MatNegGeomEFPow()),),
    # ((400, MatNegGeomEFPow()),),
    # tr inv
    # ((500, MatInv()),), # 192.15 seconds
    # ((750, MatInv()),), # 793.386 seconds
    # ((1000, MatInv()),), # nearly converged in 2000s
    # ((500, MatInvDirect()),), # slow qr in find_initial_x at /home/ptah/.julia/dev/Hypatia/src/Solvers/process.jl:122
    # ((300, MatInvDirect()),), # good, 1143.046 seconds
    # ((40, MatInvEigOrd()),), # 70.522 seconds
    # ((50, MatInvEigOrd()),), # 56 iterations and 434.485 seconds
    #
    # tr neglog
    #
    # ((500, MatNegLog()),), # 259.763 seconds
    # ((750, MatNegLog()),), # 49 iterations and 1335.334 seconds
    # ((800, MatNegLog()),), # 48 iterations and 1808.715 seconds
    # ((1000, MatNegLog()),), # far from converging in 2000s
    # ((300, MatNegLogDirect()),), # good, 1272.722 seconds
    # ((325, MatNegLogDirect()),), # 25 iterations and 1992.384 seconds
    # ((350, MatNegLogDirect()),), # far from converging after 2036.531 seconds
    # ((40, MatNegLogEigOrd()),), # 53.995 seconds
    # ((50, MatNegLogEigOrd()),), # 43 iterations and 345.392 seconds
    # ((55, MatNegLogEigOrd()),), # 40 iterations and 599.326 seconds
    ((60, MatNegLogEigOrd()),),
    #
    # tr negentropy
    #
    # ((500, MatNegEntropy()),), # 133 iterations and 750.526 seconds
    # ((600, MatNegEntropy()),), # 138 iterations and 1676.687 seconds
    ((620, MatNegEntropy()),),
    # ((750, MatNegEntropy()),), # far from converging and time limit
    # ((40, MatNegEntropyEigOrd()), nothing, (default_tol_relax = 100,)), # relaxed tols needed, 25 iterations and 44.133 seconds
    # ((50, MatNegEntropyEigOrd()), nothing, (default_tol_relax = 100,)), # 36 iterations and 300.562 seconds
    # ((55, MatNegEntropyEigOrd()), nothing, (default_tol_relax = 100,)), # slow progress, 1000 wouldn't be enough
    #
    # power
    #
    # ((500, MatPower12(1.5)),), # 135 iterations and 751.368 seconds
    # ((600, MatPower12(1.5)),), # 150 iterations and 1822.182 seconds
    # ((750, MatPower12(1.5)),), # far from converging after time limit
    ((50, MatNegLogEigOrd(1.5)),),
    ((55, MatNegLogEigOrd(1.5)),),
    ]
return (ExperimentDesignJuMP, insts)
