#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] = [
    ((2, 3, 4, false, 0, 0, 0, 0, 0),),
    ((5, 3, 4, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((5, 3, 4, false, 0.1, 0.1, 0.1, 0.2, 0.2), :SOCExpPSD),
    ((5, 4, 4, true, 0.1, 0, 0, 0, 0),),
    ((5, 3),),
    ((5, 3), :SOCExpPSD),
]
insts["fast"] = [
    ((3, 4, 5, false, 0, 0, 0, 0, 0),),
    ((3, 4, 5, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((3, 4, 5, false, 0, 0.1, 0.1, 0, 0),),
    ((20, 5),),
    ((6, 5), :SOCExpPSD),
    ((10, 20, 20, false, 0.1, 0, 0.1, 0.2, 0.2),),
    ((20, 10, 20, true, 0.1, 0, 0, 0, 0),),
    ((50, 8, 12, false, 0, 0, 0, 0, 0),),
    ((50, 8, 12, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((50, 8, 12, false, 0.1, 0.1, 0.1, 0.2, 0.2), :SOCExpPSD),
    ((15, 8, 12, true, 0.1, 0, 0, 0, 0),),
    ((15, 8, 8, true, 0, 0, 0.1, 0, 0), :SOCExpPSD),
]
insts["various"] = [
    ((3, 2, 80, true, 0.1, 0.1, 0.1, 0.2, 0.2), :SOCExpPSD),
    ((3, 2, 80, true, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((3, 2, 80, false, 0.1, 0.1, 0.1, 0.2, 0.2), :SOCExpPSD),
    ((3, 2, 80, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((6, 4, 80, true, 0.1, 0.1, 0.1, 0.2, 0.2), :SOCExpPSD),
    ((6, 4, 80, true, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((6, 4, 80, false, 0.1, 0.1, 0.1, 0.2, 0.2), :SOCExpPSD),
    ((6, 4, 80, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((12, 8, 80, true, 0.1, 0.1, 0.1, 0.2, 0.2), :SOCExpPSD),
    ((12, 8, 80, true, 0.1, 0.1, 0.1, 0.2, 0.2),),
    ((12, 8, 80, false, 0.1, 0.1, 0.1, 0.2, 0.2),),
]
return (MatrixRegressionJuMP, insts)
