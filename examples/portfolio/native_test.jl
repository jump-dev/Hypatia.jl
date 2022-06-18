#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] =
    [((4, true, false, true),), ((4, false, true, true),), ((4, false, true, false),)]
insts["fast"] = [
    ((10, true, false, true),),
    ((10, false, true, true),),
    ((10, false, true, false),),
    ((50, true, false, true),),
    ((50, false, true, true),),
    ((50, false, true, false),),
    ((200, true, false, true),),
    ((200, false, true, true),),
    ((200, false, true, false),),
]
return (PortfolioNative, insts)
