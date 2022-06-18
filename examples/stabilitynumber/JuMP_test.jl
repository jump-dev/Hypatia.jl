#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

relaxed_tols = (default_tol_relax = 100,)
insts = OrderedDict()
insts["minimal"] = [((2, true),), ((2, false),)]
insts["fast"] = [((20, true),), ((20, false),), ((30, true),), ((30, false),)]
insts["various"] = vcat(
    insts["fast"],
    [((40, true), nothing, relaxed_tols), ((40, false), nothing, relaxed_tols)],
)
return (StabilityNumber, insts)
