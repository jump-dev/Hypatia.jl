#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [((2,), :SOCExpPSD)]
insts["fast"] = [((3,), :SOCExpPSD)]
insts["various"] =
    [((3,), :SOCExpPSD), ((4,), :SOCExpPSD, relaxed_tols), ((5,), :SOCExpPSD, relaxed_tols)]
return (LotkaVolterraJuMP, insts)
