
insts = Dict()
tols7 = (tol_feas = 1e-7, tol_rel_opt = 1e-7, tol_abs_opt = 1e-7)
tols6 = (tol_feas = 1e-6, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
insts["minimal"] = [
    ((3, 2, true),),
    ((3, 2, false),),
    ]
insts["fast"] = [
    ((3, 4, true), nothing, tols7),
    ((3, 4, false), nothing, tols7),
    ((10, 15, true), nothing, tols7),
    ((10, 15, false), nothing, tols7),
    ((20, 10, true), nothing, tols6),
    ((100, 40, false), nothing, tols6),
    ]
insts["slow"] = [
    ((100, 10, true), nothing, tols7),
    ((100, 40, true), nothing, tols7),
    ]
return (ConditionNumJuMP, insts)
