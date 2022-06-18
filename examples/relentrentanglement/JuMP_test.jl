#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] = [((1, 2),), ((2, 1),)]
insts["fast"] = [((2, 3),), ((3, 3),)]
insts["various"] = [((2, 2),), ((4, 2),), ((6, 2),), ((2, 4),), ((3, 4),), ((4, 4),)]
return (RelEntrEntanglementJuMP, insts)
