#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] = [
    # nonsymmetric:
    ((false, false, false, 3, 3, 0.8),),
    ((false, false, false, 3, 3, 0.8), :ExpPSD),
    ((true, false, false, 2, 4, 0.7),),
    ((true, false, false, 2, 4, 0.7), :SOCExpPSD),
    # symmetric:
    ((false, true, false, 3, 3, 0.4),),
    ((false, true, true, 3, 3, 0.4), :ExpPSD),
    ((true, true, false, 4, 4, 0.7),),
    ((true, true, true, 4, 4, 0.7), :SOCExpPSD),
]
insts["fast"] = [
    # nonsymmetric:
    ((false, false, false, 4, 12, 0.5),),
    ((false, false, false, 4, 12, 0.5), :ExpPSD),
    ((true, false, false, 5, 10, 0.7),),
    ((true, false, false, 5, 10, 0.7), :SOCExpPSD),
    # symmetric:
    ((false, true, false, 15, 15, 0.5),),
    ((false, true, true, 15, 15, 0.5), :ExpPSD),
    ((true, true, false, 20, 20, 0.7),),
    ((true, true, true, 20, 20, 0.7), :SOCExpPSD),
]
insts["various"] = [
    # nonsymmetric:
    ((false, false, false, 20, 50, 0.8),),
    ((false, false, false, 8, 20, 0.8), :ExpPSD),
    ((true, false, false, 40, 50, 0.7),),
    ((true, false, false, 15, 20, 0.7), :SOCExpPSD),
    # symmetric:
    ((false, true, false, 100, 100, 0.9),),
    ((false, true, true, 100, 100, 0.9),),
    ((true, true, false, 50, 50, 0.5),),
    ((true, true, true, 10, 10, 0.5), :SOCExpPSD),
]
return (MatrixCompletionJuMP, insts)
