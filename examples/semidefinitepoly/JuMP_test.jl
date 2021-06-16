
relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [
    ((:matpoly2, true, true),),
    ((:matpoly5, true, true),),
    ((:matpoly5, true, false),),
    ((:matpoly5, false, false), :SOCExpPSD),
    ]
insts["fast"] = [
    ((:matpoly1, true, true),),
    ((:matpoly1, true, false),),
    ((:matpoly1, false, false), :SOCExpPSD),
    ((:matpoly2, true, true),),
    ((:matpoly2, true, false),),
    ((:matpoly2, false, false), :SOCExpPSD, relaxed_tols),
    ((:matpoly3, true, true),),
    ((:matpoly3, true, false),),
    ((:matpoly3, false, false), :SOCExpPSD),
    ((:matpoly4, true, true),),
    ((:matpoly4, true, false),),
    ((:matpoly4, false, false), :SOCExpPSD),
    ((:matpoly6, true, true),),
    ((:matpoly6, true, false),),
    ((:matpoly6, false, false), :SOCExpPSD),
    ((:matpoly7, true, true), nothing, relaxed_tols),
    ((:matpoly7, true, false),),
    ((:matpoly7, false, false), :SOCExpPSD),
    ]
insts["various"] = insts["fast"]
return (SemidefinitePolyJuMP, insts)
