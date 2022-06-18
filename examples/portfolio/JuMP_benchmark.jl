#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

portfolio_insts =
    [[(num_stocks, false, true) for num_stocks in vcat(10, 1000, 2000:2000:20000)]]

insts = OrderedDict()
insts["nat"] = (nothing, portfolio_insts)
insts["ext"] = (:SOCExpPSD, portfolio_insts)
return (PortfolioJuMP, insts)
