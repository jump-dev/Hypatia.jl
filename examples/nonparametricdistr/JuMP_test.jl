
relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [
    ((2, [VecNegLog(), VecNegGeom(), VecNeg2SqrtEF()]),),
    ((2, [VecNegLogEF(), VecInv(), VecNegEntropyEF()]),),
    ((2, [VecNegEntropy(), VecNeg2Sqrt()]),),
    ((2, [VecNegLogEF(), VecNegGeomEFExp()]),),
    ((3, [VecPower12(1.5), VecNegGeomEFPow()]),),
    ((2, [VecPower12EF(1.5), VecInvEF()]),),
    ]
insts["fast"] = [
    ((1000, [VecNegLog()]),),
    ((200, [VecNegLogEF()]), nothing, relaxed_tols),
    ((1000, [VecNegEntropy()]),),
    ((200, [VecNegLogEF()]), nothing, relaxed_tols),
    ((1000, [VecPower12(1.5)]), nothing, relaxed_tols),
    ]
insts["various"] = [
    ((1000, [VecNegLog()]),),
    ((7000, [VecNegEntropy()]),),
    ((500, [VecNegLogEF()]), nothing, relaxed_tols),
    ((1000, [VecNegLogEF()]), nothing, relaxed_tols),
    ((1000, [VecPower12(1.5)]),),
    ((500, [VecPower12EF(1.5)]), nothing, relaxed_tols),
    ((2000, [VecNegLog()]), nothing, relaxed_tols),
    ((100, [VecNegLogEF()]), nothing, relaxed_tols),
    ((400, [VecNegEntropy()]), nothing, relaxed_tols),
    ((4000, [VecNegEntropy()]), nothing, relaxed_tols),
    ]
insts["natvext"] = [
#     ((5000, 1, true),), # 28 iterations and 341.91 seconds
#     ((7000, 1, true),), # 27 iterations and 832.922 seconds
#     ((8000, 1, true),), # 27 iterations and 1218.224 seconds
#     ((15000, 1, false),), # 27 iterations and 857.543 seconds
#     ((17000, 1, false),), # 25 iterations and 1169.233 seconds
# tr neglog
    ((2000, [VecNegLogEF()]),),
    ((4000, [VecNegLogEF()]),),
    ((6000, [VecNegLogEF()]),),
    ((8000, [VecNegLogEF()]),),
    ((10000, [VecNegLogEF()]),),
    ((2000, [VecNegLog()]),),
    ((4000, [VecNegLog()]),),
    ((6000, [VecNegLog()]),),
    ((8000, [VecNegLog()]),),
    ((10000, [VecNegLog()]),),
    ((12000, [VecNegLog()]),),
    ((14000, [VecNegLog()]),),
    ((16000, [VecNegLog()]),),
    ((18000, [VecNegLog()]),),
    # tr negentropy
    ((2000, [VecNegEntropyEF()]),),
    ((4000, [VecNegEntropyEF()]),),
    ((6000, [VecNegEntropyEF()]),),
    ((8000, [VecNegEntropyEF()]),),
    ((10000, [VecNegEntropyEF()]),),
    ((2000, [VecNegEntropy()]),),
    ((4000, [VecNegEntropy()]),),
    ((6000, [VecNegEntropy()]),),
    ((8000, [VecNegEntropy()]),),
    ((10000, [VecNegEntropy()]),),
    ((12000, [VecNegEntropy()]),),
    ((14000, [VecNegEntropy()]),),
    ((16000, [VecNegEntropy()]),),
    ((18000, [VecNegEntropy()]),),
    # tr power12
    ((2000, [VecPower12EF(1.5)]),),
    ((4000, [VecPower12EF(1.5)]),),
    ((6000, [VecPower12EF(1.5)]),),
    ((8000, [VecPower12EF(1.5)]),),
    ((10000, [VecPower12EF(1.5)]),),
    ((2000, [VecPower12(1.5)]),),
    ((4000, [VecPower12(1.5)]),),
    ((6000, [VecPower12(1.5)]),),
    ((8000, [VecPower12(1.5)]),),
    ((10000, [VecPower12(1.5)]),),
    ((12000, [VecPower12(1.5)]),),
    ((14000, [VecPower12(1.5)]),),
    ((16000, [VecPower12(1.5)]),),
    ((18000, [VecPower12(1.5)]),),
    ]
return (NonparametricDistrJuMP, insts)
