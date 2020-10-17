
relaxed_tols = (default_tol_relax = 100,)
insts = Dict()
insts["minimal"] = [
    ((2, 3),),
    ((2, 3), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((5, 10), nothing, relaxed_tols),
    ((5, 10), StandardConeOptimizer, relaxed_tols),
    ((10, 20), nothing, relaxed_tols),
    ((10, 20), StandardConeOptimizer, relaxed_tols),
    ((20, 40), nothing, relaxed_tols),
    ((20, 40), StandardConeOptimizer, relaxed_tols),
    ((40, 80), nothing, relaxed_tols),
    ((40, 80), StandardConeOptimizer, relaxed_tols),
    ((100, 150), nothing, relaxed_tols),
    ((100, 150), StandardConeOptimizer, relaxed_tols),
    ]
insts["slow"] = [
    ((40, 80), StandardConeOptimizer, relaxed_tols),
    ((100, 200), nothing, relaxed_tols),
    ]
return (RobustGeomProgJuMP, insts)
