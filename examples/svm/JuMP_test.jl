
relaxed_tols = (default_tol_relax = 1000,)
insts = Dict()
insts["minimal"] = [
    ((2, 1, true, true),)
    ((2, 1, false, true), nothing, relaxed_tols)
    ]

insts["fast"] = [
    ((100, 100, true, true),)
    ((100, 100, false, true), nothing, relaxed_tols)
    ((20, 100, true, true),)
    ((20, 100, false, true), nothing, relaxed_tols)
    ((20, 200, true, true),)
    ((20, 200, false, true), nothing, relaxed_tols)
    ((20, 2000, true, true),)
    ((20, 2000, false, true), nothing, relaxed_tols)
    ]

return (SVMJuMP, insts)
