
insts = Dict()
tols6 = (tol_feas = 1e-6, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
insts["minimal"] = [
    ((2, 3),),
    ((2, 3), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((5, 10), nothing, tols6),
    ((5, 10), StandardConeOptimizer, tols6),
    ((10, 20), nothing, tols6),
    ((10, 20), StandardConeOptimizer, tols6),
    ((20, 40), nothing, tols6),
    ((20, 40), StandardConeOptimizer, tols6),
    ((40, 80), nothing, tols6),
    ((40, 80), StandardConeOptimizer, tols6),
    ((100, 150), nothing, tols6),
    ((100, 150), StandardConeOptimizer, tols6),
    ]
insts["slow"] = [
    ((40, 80), StandardConeOptimizer, tols6),
    ((100, 200), nothing, tols6),
    ]
return (RobustGeomProgJuMP, insts)
