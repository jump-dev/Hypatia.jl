
portfolio_insts = [
    [(num_stocks, false, true) for num_stocks in vcat(10, 500:500:2000, 3000:1000:13000)] # includes compile run
    ]

insts = Dict()
insts["nat"] = (nothing, portfolio_insts)
insts["ext"] = (SOCExpPSDOptimizer, portfolio_insts)
return (PortfolioJuMP, insts)
