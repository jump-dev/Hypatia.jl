#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] = [
    # tr neglog
    ((3, false, MatLogdetCone()),),
    ((3, true, MatLogdetCone()),),
    ((3, false, MatNegLog()),),
    ((3, true, MatNegLog()),),
    ((3, false, MatNegLogDirect()),),
    ((3, true, MatNegLogDirect()),),
    ((3, false, MatNegLogEigOrd()),),
    # negrtdet
    ((3, false, MatNegRtdet()),),
    ((3, true, MatNegRtdet()),),
    ((3, false, MatNegRtdetEFExp()),),
    ((3, true, MatNegRtdetEFExp()),),
    ((3, false, MatNegRtdetEFPow()),),
    ((3, true, MatNegRtdetEFPow()),),
    # tr negsqrtconj
    ((3, false, MatNegSqrtConj()),),
    ((3, false, MatNegSqrtConjDirect()),),
    ((3, false, MatNegSqrtConjEigOrd()),),
    ((3, true, MatNegSqrtConj()),),
    ((3, true, MatNegSqrtConjDirect()),),
    ((3, true, MatNegSqrtConjEigOrd()),),
    # tr negentropy
    ((3, false, MatNegEntropy()),),
    ((3, false, MatNegEntropyEigOrd()),),
    # tr negsqrt
    ((3, false, MatNegSqrt()),),
    ((3, false, MatNegSqrtEigOrd()),),
    # tr negpower01
    ((3, false, MatNegPower01(0.7)),),
    ((3, false, MatNegPower01(0.7)),),
    # tr power12
    ((3, false, MatPower12(1.3)),),
    ((3, false, MatPower12EigOrd(1.3)),),
]
insts["fast"] = [
    ((30, true, MatNegRtdet()),),
    ((15, false, MatNegRtdetEFExp()),),
    ((15, false, MatNegRtdetEFPow()),),
    ((40, false, MatNegSqrtConj()),),
    ((8, true, MatNegSqrtConjEigOrd()),),
    ((20, false, MatNegSqrtConjDirect()),),
    ((15, true, MatNegSqrtConjDirect()),),
    ((50, false, MatLogdetCone()),),
    ((30, false, MatNegLog()),),
    ((20, true, MatNegLog()),),
    ((6, false, MatNegLogEigOrd()),),
    ((15, false, MatNegLogDirect()),),
    ((10, true, MatNegLogDirect()),),
    ((30, false, MatNegEntropy()),),
    ((8, true, MatNegEntropyEigOrd()),),
    ((30, false, MatPower12(1.5)),),
    ((7, false, MatPower12EigOrd(1.5)),),
]
insts["various"] = [
    ((100, false, MatNegRtdet()),),
    ((60, true, MatNegRtdet()),),
    ((50, false, MatNegRtdetEFExp()),),
    ((30, true, MatNegRtdetEFPow()),),
    ((100, false, MatNegSqrtConj()),),
    ((12, false, MatNegSqrtConjEigOrd()),),
    ((50, false, MatNegSqrtConjDirect()),),
    ((30, true, MatNegSqrtConjDirect()),),
    ((150, false, MatLogdetCone()),),
    ((80, true, MatLogdetCone()),),
    ((100, false, MatNegLog()),),
    ((50, true, MatNegLog()),),
    ((18, false, MatNegLogEigOrd()), nothing, (default_tol_relax = 1000,)),
    ((80, false, MatNegLogDirect()),),
    ((40, true, MatNegLogDirect()),),
    ((75, false, MatNegEntropy()),),
    ((14, false, MatNegEntropyEigOrd()),),
    ((30, false, MatPower12(1.5)),),
    ((6, true, MatPower12EigOrd(1.5)),),
]
return (CovarianceEstJuMP, insts)
