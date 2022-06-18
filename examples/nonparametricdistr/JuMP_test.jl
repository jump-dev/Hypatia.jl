#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [
    # geomean
    ((3, VecNegRtdet()),),
    ((3, VecNegRtdetEFExp()),),
    ((3, VecNegRtdetEFPow()),),
    # sum neglog
    ((2, VecLogCone()),),
    ((2, VecNegLog()),),
    ((2, VecNegLogEF()),),
    # sum negentropy
    ((2, VecNegEntropy()),),
    ((2, VecNegEntropyEF()),),
    # sum negsqrt
    ((2, VecNegSqrt()),),
    ((2, VecNegSqrtEF()),),
    # sum negpower01
    ((2, VecNegPower01(0.5)),),
    ((2, VecNegPower01EF(0.5)),),
    # sum power12
    ((2, VecPower12(1.5)),),
    ((2, VecPower12EF(1.5)),),
    # mixture
    ((2, VecNegRtdet()),),
    ((3, VecPower12(1.5)),),
    ((3, VecNegLogEF()),),
]
insts["fast"] = [
    ((1000, VecNegLog()),),
    ((500, VecLogCone()),),
    ((200, VecNegLogEF()), nothing, relaxed_tols),
    ((1000, VecNegEntropy()),),
    ((200, VecNegLogEF()), nothing, relaxed_tols),
    ((1000, VecNegPower01(0.3)), nothing, relaxed_tols),
    ((1000, VecPower12(1.5)), nothing, relaxed_tols),
]
insts["various"] = [
    ((1000, VecNegLog()),),
    ((3000, VecLogCone()),),
    ((7000, VecNegEntropy()),),
    ((500, VecNegLogEF()), nothing, relaxed_tols),
    ((1000, VecNegLogEF()), nothing, relaxed_tols),
    ((1000, VecNegSqrt()), nothing, relaxed_tols),
    ((500, VecNegSqrtEF()), nothing, relaxed_tols),
    ((500, VecNegSqrtConj()), nothing, relaxed_tols),
    ((500, VecNegSqrtConjEF()), nothing, relaxed_tols),
    ((1000, VecNegPower01(0.4)),),
    ((500, VecNegPower01EF(0.4)), nothing, relaxed_tols),
    ((1000, VecPower12(1.5)),),
    ((500, VecPower12EF(1.5)), nothing, relaxed_tols),
    ((2000, VecNegLog()), nothing, relaxed_tols),
    ((100, VecNegLogEF()), nothing, relaxed_tols),
    ((400, VecNegEntropy()), nothing, relaxed_tols),
    ((4000, VecNegEntropy()), nothing, relaxed_tols),
]
return (NonparametricDistrJuMP, insts)
