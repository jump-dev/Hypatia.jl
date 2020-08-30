
relaxed_tols = (init_tol_qr = 1e-8,)
insts = Dict()
insts["minimal"] = [
    ((3,), nothing, relaxed_tols),
    ]
insts["fast"] = [
    ((4,),),
    ]
insts["slow"] = [
    ((6,),),
    ]
return (LotkaVolterraJuMP, insts)
