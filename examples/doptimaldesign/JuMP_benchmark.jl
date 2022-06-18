#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

function doptimaldesign_insts(use_logdet::Bool)
    return [[
        (q, 2q, 2q, 5, use_logdet, !use_logdet, false) for
        q in vcat(3, 50:50:200, 300:100:1000)
    ]]
end

insts = OrderedDict()
rootdet_insts = doptimaldesign_insts(false)
all_insts = vcat(rootdet_insts, doptimaldesign_insts(true))
insts["nat"] = (nothing, all_insts)
insts["extEP"] = (:ExpPSD, all_insts)
insts["extSEP"] = (:SOCExpPSD, rootdet_insts)
return (DOptimalDesignJuMP, insts)
