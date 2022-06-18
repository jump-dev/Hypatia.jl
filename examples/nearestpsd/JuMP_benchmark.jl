#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

function nearestpsd_insts(use_nat::Bool)
    return [
        [
            (side, use_completable, false, use_nat, true) for
            side in vcat(30, 50:50:200, 300:100:800)
        ] # includes compile run
        for use_completable in (false, true)
    ]
end

insts = OrderedDict()
insts["nat"] = (nothing, nearestpsd_insts(true))
insts["ext"] = (:SOCExpPSD, nearestpsd_insts(false))
return (NearestPSDJuMP, insts)
