
portfolio_insts = [
    [(num_stocks, false, true)
    for num_stocks in vcat(10, 1000, 2000:2000:20000)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, portfolio_insts)
insts["ext"] = (:SOCExpPSD, portfolio_insts)
return (PortfolioJuMP, insts)
