#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] =
    [((3, false, false),), ((3, false, true),), ((3, true, false),), ((3, true, true),)]
insts["fast"] = [
    ((20, false, false),),
    ((20, true, false),),
    ((8, false, true),),
    ((5, true, true),),
    ((50, false, false),),
    ((50, true, false),),
    ((12, false, true),),
    ((8, true, true),),
]
insts["various"] = [
    ((100, false, false),),
    ((100, true, false),),
    ((12, false, true),),
    ((8, true, true),),
    ((200, false, false),),
    ((200, true, false),),
    ((15, false, true), nothing, (default_tol_relax = 100,)),
    ((300, false, false),),
    ((250, true, false),),
    ((17, false, true), nothing, (default_tol_relax = 1000,)),
]
return (ClassicalQuantum, insts)
