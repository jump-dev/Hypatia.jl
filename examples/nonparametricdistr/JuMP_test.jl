
relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [
    # tr neglog
    ((2, 2, VecNegLog()),),
    ((2, 2, VecNegLogEF()),),
    # tr negentropy
    ((2, 2, VecNegEntropy()),),
    ((2, 2, VecNegLogEF()),),
    # tr power12
    ((2, 2, VecPower12(1.5)),),
    ((2, 2, VecPower12EF(1.5)),),
    ]
insts["fast"] = [
    ((1000, 1, VecNegLog()),),
    ((200, 1, VecNegLogEF()), nothing, relaxed_tols),
    ((1000, 2, VecNegEntropy()),),
    ((200, 2, VecNegLogEF()), nothing, relaxed_tols),
    ((1000, 3, VecPower12(1.5)), nothing, relaxed_tols),
    ]
insts["various"] = [
    ((1000, 1, VecNegLog()),),
    ((7000, 1, VecNegEntropy()),),
    ((500, 1, VecNegLogEF()), nothing, relaxed_tols),
    ((1000, 1, VecNegLogEF()), nothing, relaxed_tols),
    ((1000, 2, VecPower12(1.5)),),
    ((500, 2, VecPower12EF(1.5)), nothing, relaxed_tols),
    ((2000, 3, VecNegLog()), nothing, relaxed_tols),
    ((100, 3, VecNegLogEF()), nothing, relaxed_tols),
    ((400, 3, VecNegEntropy()), nothing, relaxed_tols),
    ((4000, 3, VecNegEntropy()), nothing, relaxed_tols),
    ]
insts["natvext"] = [
    ((5000, 1, true),), # 28 iterations and 341.91 seconds
    ((7000, 1, true),), # 27 iterations and 832.922 seconds
    ((8000, 1, true),), # 27 iterations and 1218.224 seconds
    ((15000, 1, false),), # 27 iterations and 857.543 seconds
    ((17000, 1, false),), # 25 iterations and 1169.233 seconds
    ]
