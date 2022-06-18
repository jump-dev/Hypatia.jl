#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

relaxed_infeas_tols = (tol_infeas = 1e-9,)
insts = OrderedDict()
insts["minimal"] = [
    ((0.7, 4, 1e-3, true, true),),
    ((0.7, 4, 1e-3, false, true), :SOCExpPSD),
    ((2.0, 4, 1e-3, true, false), nothing, relaxed_infeas_tols),
    ((2.0, 4, 1e-3, false, false), :SOCExpPSD, relaxed_infeas_tols),
]
insts["fast"] = [
    ((1.0, 2, 1e-3, true, false),),
    ((1.0, 2, 1e-3, false, false), :SOCExpPSD),
    ((4.0, 2, 1e-3, true, false),),
    ((4.0, 2, 1e-3, false, false), :SOCExpPSD),
]
insts["various"] = vcat(insts["minimal"], insts["fast"])
return (ContractionJuMP, insts)
