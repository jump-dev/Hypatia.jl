#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] = [
    ((2, 3, 4, 2, true, false, false, true, true, false),),
    ((2, 3, 4, 2, true, false, false, true, true, true),),
    ((2, 3, 4, 2, false, true, false, true, true, true),),
    ((2, 3, 4, 2, false, false, true, true, true, true),),
    ((2, 3, 4, 2, true, false, false, false, false, true),),
    ((2, 3, 4, 2, false, true, false, false, false, true),),
]
insts["fast"] = [
    ((3, 5, 7, 2, true, false, false, true, true, false),),
    ((3, 5, 7, 2, true, false, false, true, true, true),),
    ((3, 5, 7, 2, false, true, false, true, true, true),),
    ((3, 5, 7, 2, false, false, true, true, true, true),),
    ((3, 5, 7, 2, true, false, false, false, false, true),),
    ((3, 5, 7, 2, false, true, false, false, false, true),),
    ((8, 15, 25, 5, true, false, false, true, true, false),),
    ((8, 15, 25, 5, true, false, false, true, true, true),),
    ((8, 15, 25, 5, false, true, false, true, true, true),),
    ((8, 15, 25, 5, false, false, true, true, true, true),),
    ((8, 15, 25, 5, true, false, false, false, false, true),),
    ((8, 15, 25, 5, false, true, false, false, false, true),),
    ((25, 75, 125, 5, true, false, false, true, true, false),),
    ((25, 75, 125, 5, true, false, false, true, true, true),),
    ((25, 75, 125, 5, false, true, false, true, true, true),),
    ((25, 75, 125, 5, false, false, true, true, true, true),),
    ((25, 75, 125, 5, true, false, false, false, false, true),),
    ((25, 75, 125, 5, false, true, false, false, false, true),),
]
return (DOptimalDesignNative, insts)
