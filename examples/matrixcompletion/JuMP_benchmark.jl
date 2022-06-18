#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

matrixcompletion_insts = [
    [(false, false, false, d, k * d, 0.8) for d in vcat(2, 5:5:max_d)] # includes compile run
    for (k, max_d) in ((10, 45), (20, 30))
]

insts = OrderedDict()
insts["nat"] = (nothing, matrixcompletion_insts)
insts["extEP"] = (:ExpPSD, matrixcompletion_insts)
insts["extSEP"] = (:SOCExpPSD, matrixcompletion_insts)
return (MatrixCompletionJuMP, insts)
