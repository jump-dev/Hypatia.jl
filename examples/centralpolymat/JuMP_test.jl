
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
insts["natvext"] = [
    # tr inv
    # ((8, 3, MatInv()),), # good, 1066.2s
    # ((6, 6, MatInv()),), # memory error in densify
    # ((4, 12, MatInv()),),  # memory error in densify
    # ((6, 4, MatInv()),), # v far from converging after 2000s
    # ((4, 6, MatInv()),), # v far from converging after 2000s
    # ((8, 4, MatInvDirect()),), # memory error in densify
    # ((6, 6, MatInvDirect()),), # memory error in densify
    # ((2, 6, MatInvEigOrd()),), # good, 982.957
    # ((4, 4, MatInvEigOrd()),), # memory error in densify
    # neg entr
    # ((2, 10, MatNegEntropy()),), # 25 iterations and 37.121 seconds
    # ((2, 11, MatNegEntropy()),), # 26 iterations and 60.033 seconds
    ((2, 13, MatNegEntropy()),),
    # ((8, 3, MatNegEntropy()),), # good, 1591.051
    # ((4, 4, MatNegEntropy()),), # 27 iterations and 41.804 seconds
    # ((4, 5, MatNegEntropy()),), # 28 iterations and 465.674 seconds
    # ((2, 6, MatNegEntropyEigOrd()), nothing, relaxed_tols), # relaxed tols needed, 22 iterations and 711.949 seconds
    # ((2, 7, MatNegEntropyEigOrd()), nothing, relaxed_tols), # too big, tl in 0 iters
    # ((4, 3, MatNegEntropyEigOrd()),), too big, tl in 0 iters
    # ((3, 3, MatNegEntropyEigOrd()), nothing, relaxed_tols), # relaxed tols needed, 18 iterations and 49.083 seconds
    # ((1, 10, MatNegEntropyEigOrd()),), # 18 iterations and 0.976 second
    # ((1, 20, MatNegEntropyEigOrd()),), # 29 iterations and 95.061 seconds
    ((1, 30, MatNegEntropyEigOrd()),),
    # for ord can go as big as (2, 6), (3, 3), symmetric wrt n, d so (4, 2) then stop
    # for nat (3, 8), (4, 5)
    #
    # power
    # ((2, 9, MatPower12(1.5)), nothing, relaxed_tols), # 23 iterations and 6.617 seconds
    # ((2, 11, MatPower12(1.5)), nothing, relaxed_tols), # 34 iterations and 51.489 seconds
    ((2, 13, MatPower12(1.5)), nothing, relaxed_tols),
    # ((8, 2, MatPower12(1.5)), nothing, relaxed_tols), # 18 iterations and 1.353 seconds
    # ((7, 3, MatPower12(1.5)), nothing, relaxed_tols), # 29 iterations and 289.864 seconds
    # ((8, 3, MatPower12(1.5)), nothing, relaxed_tols), # relaxed tols needed, hit 2028.072 seconds and was struggling but ok viols like 1e-5, 1e-6
    # ((4, 4, MatPower12(1.5)),), # 31 iterations and 24.411 seconds
    # ((4, 5, MatPower12(1.5)),), # 40 iterations and 617.041 seconds
    # ((2, 6, MatPower12EigOrd(1.5)), nothing, relaxed_tols), # relaxed tols needed, 32 iterations and 866.97 seconds
    # ((3, 3, MatPower12EigOrd(1.5)), nothing, relaxed_tols), # relaxed tols needed, 19 iterations and 49.388 seconds
    # MatPower12EigOrd can go as big as (2, 6), (3, 3)
    #
    # neg log
    # ((2, 10, MatNegLog()),), # 59 iterations and 38.681 seconds
    # ((2, 11, MatNegLog()),), # 70 iterations and 103.35 seconds
    # ((8, 3, MatNegLog()),), # 18 iterations and 1002.817 seconds
    # ((4, 4, MatNegLog()),), # 23 iterations and 18.455 seconds
    # ((4, 5, MatNegLog()),), # 32 iterations and 502.311 seconds
    # ((8, 2, MatNegLogDirect()), nothing, relaxed_tols), # relaxed tols needed, 14 iterations and 3.254 seconds
    # ((2, 6, MatNegLogEigOrd()),), # 50 iterations and 1138.664 seconds
    ]
return (CentralPolyMatJuMP, insts)
