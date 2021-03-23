
relaxed_tols = (default_tol_relax = 100,)
insts = Dict()
insts["minimal"] = [
    ((:matpoly2, true, true),),
    ((:matpoly5, true, true),),
    ((:matpoly5, true, false),),
    ((:matpoly5, false, false), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    ((:matpoly1, true, true), nothing, relaxed_tols),
    ((:matpoly1, true, false),),
    ((:matpoly1, false, false), SOCExpPSDOptimizer),
    ((:matpoly2, true, true),),
    ((:matpoly2, true, false),),
    ((:matpoly2, false, false), SOCExpPSDOptimizer, relaxed_tols),
    ((:matpoly3, true, true), nothing, relaxed_tols),
    ((:matpoly3, true, false), nothing, relaxed_tols),
    ((:matpoly3, false, false), SOCExpPSDOptimizer),
    ((:matpoly4, true, true), nothing, relaxed_tols),
    ((:matpoly4, true, false),),
    ((:matpoly4, false, false), SOCExpPSDOptimizer),
    ((:matpoly6, true, true),),
    ((:matpoly6, true, false),),
    ((:matpoly6, false, false), SOCExpPSDOptimizer),
    ((:matpoly7, true, true), nothing, relaxed_tols),
    ((:matpoly7, true, false), nothing, relaxed_tols),
    ((:matpoly7, false, false), SOCExpPSDOptimizer),
    ]
insts["slow"] = Tuple[]
insts["various"] = insts["fast"]
return (SemidefinitePolyJuMP, insts)
