#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

matrixregression_insts = [
    vcat((4, 3), [(n, m) for n in ns]) # includes compile run
    for (m, ns) in
    ((15, vcat(50:50:150, 250, 500:500:3000)), (30, vcat(50:50:150, 250:250:1500)))
]

insts = OrderedDict()
insts["nat"] = (nothing, matrixregression_insts)
insts["ext"] = (:SOCExpPSD, matrixregression_insts)
return (MatrixRegressionJuMP, insts)
