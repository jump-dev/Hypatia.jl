
portfolio_insts = [
    [(num_stocks, false, true)
    for num_stocks in vcat(10, 500, 1000:1000:16000)] # includes compile run
    ]

insts = OrderedDict()
insts["nat"] = (nothing, portfolio_insts)
insts["ext"] = (:SOCExpPSD, portfolio_insts)
return (PortfolioJuMP, insts)
