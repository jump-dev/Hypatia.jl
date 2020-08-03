
portfolio_instances = [
    [(num_stocks, false, true) for num_stocks in vcat(10, 500:500:11000)] # includes compile run
    ]

instances[PortfolioJuMP]["nat"] = (nothing, portfolio_instances)
instances[PortfolioJuMP]["ext"] = (StandardConeOptimizer, portfolio_instances)
