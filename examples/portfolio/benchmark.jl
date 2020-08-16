
portfolio_insts = [
    [(num_stocks, false, true) for num_stocks in vcat(10, 500:500:11000)] # includes compile run
    ]

insts[PortfolioJuMP]["nat"] = (nothing, portfolio_insts)
insts[PortfolioJuMP]["ext"] = (StandardConeOptimizer, portfolio_insts)
