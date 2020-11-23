
# matrixregression_insts = [
#     [(k, d) for d in vcat(2, range_d)] # includes compile run
#     # for (k, range_d) in ((10, 10:10:50), (50, 5:5:25))
#     for (k, range_d) in ((10, 10:10:10), (50, 5:5:5))
#     ]

# matrixregression_insts = [
#     vcat((2, 3), [(2^pow_k, d) for pow_k in range_pow_k]) # includes compile run
#     for (d, range_pow_k) in ((5, 1:1:9), (10, 1:1:7), (20, 0:1:6))
#     ]

matrixregression_insts = [
    vcat((4, 3), [(n, m) for n in ns]) # includes compile run
    for (m, ns) in ((15, 100:100:1000), (30, 50:50:500))
    ]


insts = Dict()
insts["nat"] = (nothing, matrixregression_insts)
insts["ext"] = (SOCExpPSDOptimizer, matrixregression_insts)
return (MatrixRegressionJuMP, insts)
