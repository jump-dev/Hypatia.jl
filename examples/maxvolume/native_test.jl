#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] =
    [((4, true, false, false),), ((4, false, true, false),), ((4, false, false, true),)]
insts["fast"] = [
    ((10, true, false, false),),
    ((10, false, true, false),),
    ((10, false, false, true),),
    ((100, true, false, false),),
    ((100, false, true, false), (default_tol_relax = 100,)),
    ((100, false, false, true),),
    ((1000, true, false, false),),
]
insts["various"] = [
    ((750, true, false, false),),
    ((750, false, false, true),),
    ((1500, true, false, false),),
    ((1500, false, false, true),),
]
return (MaxVolumeNative, insts)
