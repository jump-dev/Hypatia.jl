#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

insts = OrderedDict()
insts["minimal"] = [((2,),)]
insts["fast"] = [((5,),), ((10,),)]
insts["various"] = [((5,),), ((10,),), ((15,),)]
return (NearestCorrelationJuMP, insts)
