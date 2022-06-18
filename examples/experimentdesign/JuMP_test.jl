#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] = [
    # rootdet
    ((3, MatNegRtdet()),),
    ((3, MatNegRtdetEFExp()),),
    ((3, MatNegRtdetEFPow()),),
    # tr neglog
    ((2, MatLogdetCone()),),
    ((2, MatNegLog()),),
    ((2, MatNegLogDirect()),),
    # tr negentropy
    ((3, MatNegEntropy()),),
    ((3, MatNegEntropyEigOrd()),),
    # tr negsqrt
    ((3, MatNegSqrt()),),
    ((3, MatNegSqrtEigOrd()),),
    # tr negpower01
    ((3, MatNegPower01(0.7)),),
    ((3, MatNegPower01EigOrd(0.7)),),
    # tr power12
    ((3, MatPower12(1.3)),),
    ((3, MatPower12EigOrd(1.3)),),
]
insts["fast"] = [
    ((50, MatNegRtdet()),),
    ((12, MatNegRtdetEFExp()),),
    ((8, MatNegRtdetEFPow()),),
    ((40, MatNegSqrt()),),
    ((5, MatNegSqrtEigOrd()),),
    ((6, MatNegSqrtConjEigOrd()),),
    ((25, MatNegSqrtConjDirect()),),
    ((100, MatLogdetCone()),),
    ((75, MatNegLog()),),
    ((7, MatNegLogEigOrd()),),
    ((20, MatNegLogDirect()),),
    ((60, MatNegEntropy()),),
    ((8, MatNegEntropyEigOrd()),),
    ((50, MatPower12(1.5)),),
    ((5, MatPower12EigOrd(1.5)),),
    ((50, MatNegPower01(0.4)),),
    ((40, MatNegPower01Conj(0.4)),),
    ((5, MatNegPower01EigOrd(0.4)),),
]
insts["various"] = [
    ((400, MatNegRtdet()),),
    ((70, MatNegRtdetEFExp()),),
    ((40, MatNegRtdetEFPow()), nothing, (default_tol_relax = 100,)),
    ((200, MatNegSqrt()),),
    ((10, MatNegSqrtConjEigOrd()),),
    ((125, MatNegSqrtConjDirect()),),
    ((300, MatLogdetCone()),),
    ((150, MatNegLog()),),
    ((12, MatNegLogEigOrd()),),
    ((75, MatNegLogDirect()),),
    ((150, MatNegEntropy()),),
    ((10, MatNegEntropyEigOrd()),),
    ((80, MatNegPower01(0.4)),),
    ((40, MatNegPower01Conj(0.6)),),
    ((15, MatPower12(1.5)),),
    ((9, MatPower12EigOrd(1.5)), nothing, (default_tol_relax = 100,)),
]
return (ExperimentDesignJuMP, insts)
