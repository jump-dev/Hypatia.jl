
insts = Dict()
insts["minimal"] = [
    ((:matpoly2, true, true),),
    ((:matpoly5, true, true),),
    ((:matpoly5, true, false),),
    ((:matpoly5, false, false),),
    ]
insts["fast"] = [
    ((:matpoly1, true, true),),
    ((:matpoly1, true, false),),
    ((:matpoly1, false, false),),
    ((:matpoly2, true, true),),
    ((:matpoly2, true, false),),
    ((:matpoly2, false, false),),
    ((:matpoly3, true, true),),
    ((:matpoly3, true, false),),
    ((:matpoly3, false, false),),
    ((:matpoly4, true, true),),
    ((:matpoly4, true, false),),
    ((:matpoly4, false, false),),
    ((:matpoly6, true, true),),
    ((:matpoly6, true, false),),
    ((:matpoly6, false, false),),
    ((:matpoly7, true, true),),
    ((:matpoly7, true, false),),
    ((:matpoly7, false, false),),
    ]
insts["slow"] = Tuple[]
return (SemidefinitePolyJuMP, insts)
