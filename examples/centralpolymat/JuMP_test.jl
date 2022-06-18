#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

relaxed_tols = (default_tol_relax = 100,)
insts = OrderedDict()
insts["minimal"] = [
    # rootdet
    ((2, 2, MatNegRtdet()),),
    ((2, 2, MatNegRtdetEFExp()),),
    ((2, 2, MatNegRtdetEFPow()),),
    # tr negentropyconj (domain not positive)
    ((1, 2, MatNegEntropyConj()),),
    ((1, 2, MatNegEntropyConjEigOrd()),),
    # tr negsqrtconj
    ((1, 2, MatNegSqrtConj()),),
    ((1, 2, MatNegSqrtConjDirect()),),
    # tr negpower01conj
    ((1, 2, MatNegPower01Conj(0.3)),),
    ((1, 2, MatNegPower01ConjEigOrd(0.3)), nothing, relaxed_tols),
    # tr power12conj (domain not positive)
    ((1, 2, MatPower12Conj(1.7)),),
    ((1, 2, MatPower12ConjEigOrd(1.7)),),
]
insts["fast"] = [
    ((1, 10, MatNegRtdetEFExp()),),
    ((1, 15, MatNegRtdet()),),
    ((2, 3, MatNegRtdetEFPow()),),
    ((2, 3, MatNegSqrtEigOrd()),),
    ((2, 6, MatNegSqrt()),),
    ((2, 5, MatNegSqrtConjDirect()),),
    ((2, 6, MatNegSqrtConj()),),
    ((2, 6, MatNegEntropy()),),
    ((2, 7, MatLogdetCone()),),
    ((3, 2, MatNegEntropyEigOrd()),),
    ((3, 2, MatNegEntropyConj()),),
    ((3, 2, MatNegEntropyConjEigOrd()),),
    ((3, 2, MatNegPower01Conj(0.3)),),
    ((3, 2, MatNegPower01ConjEigOrd(0.3)),),
    ((3, 2, MatPower12EigOrd(1.5)),),
    ((3, 4, MatPower12(1.5)),),
    ((3, 4, MatNegRtdet()),),
    ((7, 2, MatNegLog()),),
    ((7, 2, MatPower12(1.5)),),
    ((7, 2, MatNegPower01(0.7)),),
]
insts["various"] = [
    ((2, 5, MatNegSqrtEigOrd()),),
    ((2, 5, MatNegLogEigOrd()),),
    ((2, 4, MatPower12EigOrd(1.5)), nothing, relaxed_tols),
    ((2, 4, MatNegEntropyEigOrd()),),
    ((2, 10, MatNegRtdetEFExp()),),
    ((2, 8, MatNegRtdetEFPow()),),
    ((2, 14, MatNegRtdet()),),
    ((2, 12, MatNegSqrtConj()),),
    ((2, 10, MatLogdetCone()),),
    ((2, 8, MatNegEntropy()),),
    ((2, 6, MatNegPower01(0.7)),),
    ((3, 5, MatNegLogDirect()),),
    ((3, 6, MatNegEntropy()),),
    ((3, 6, MatNegLog()),),
    ((3, 5, MatPower12(1.5)),),
    ((3, 5, MatNegEntropyConj()),),
    ((3, 4, MatNegPower01Conj(0.3)),),
    ((5, 2, MatNegEntropyConjEigOrd()), nothing, relaxed_tols),
    ((5, 2, MatPower12ConjEigOrd(1.5)),),
    ((5, 3, MatNegSqrtConjDirect()),),
    ((5, 3, MatNegLogDirect()),),
    ((6, 3, MatNegRtdet()),),
    ((6, 3, MatNegSqrt()),),
]
return (CentralPolyMatJuMP, insts)
