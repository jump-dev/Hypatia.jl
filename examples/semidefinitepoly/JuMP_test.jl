
insts = Dict()
insts["minimal"] = [
    ((:matpoly2, true, true),),
    ((:matpoly5, true, true),),
    ((:matpoly5, true, false),),
    ((:matpoly5, false, false), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((:matpoly1, true, true),),
    ((:matpoly1, true, false),),
    ((:matpoly1, false, false), StandardConeOptimizer),
    ((:matpoly2, true, true),),
    ((:matpoly2, true, false),),
    ((:matpoly2, false, false), StandardConeOptimizer),
    ((:matpoly3, true, true),),
    ((:matpoly3, true, false),),
    ((:matpoly3, false, false), StandardConeOptimizer),
    ((:matpoly4, true, true),),
    ((:matpoly4, true, false),),
    ((:matpoly4, false, false), StandardConeOptimizer),
    ((:matpoly6, true, true),),
    ((:matpoly6, true, false),),
    ((:matpoly6, false, false), StandardConeOptimizer),
    ((:matpoly7, true, true),),
    ((:matpoly7, true, false),),
    ((:matpoly7, false, false), StandardConeOptimizer),
    ]
insts["slow"] = Tuple[]
return (SemidefinitePolyJuMP, insts)
