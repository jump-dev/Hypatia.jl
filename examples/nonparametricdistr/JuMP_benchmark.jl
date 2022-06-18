#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

function nonparametricdistr_insts(ext::VecSpecExt)
    return [[(d, ext) for d in vcat(10, 500, 1000, 2500, 5000:5000:30000)]]
end

insts = OrderedDict()
insts["nat"] = (
    nothing,
    vcat(
        nonparametricdistr_insts.([
            VecNegRtdet(),
            VecNegLog(),
            VecNegSqrt(),
            VecNegEntropy(),
        ])...,
    ),
)
insts["vecext"] = (
    nothing,
    vcat(
        nonparametricdistr_insts.([
            VecNegRtdetEFExp(),
            VecNegLogEF(),
            VecNegSqrtEF(),
            VecNegEntropyEF(),
        ])...,
    ),
)
return (NonparametricDistrJuMP, insts)
