#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] = [((4, true),), ((4, false), :SOCExpPSD)]
insts["fast"] = [((6, true),), ((6, false), :SOCExpPSD), ((8, true),)]
insts["various"] = vcat(
    insts["fast"],
    [((8, false), :SOCExpPSD), ((10, true),), ((10, false), :SOCExpPSD)],
)
return (RegionOfAttrJuMP, insts)
