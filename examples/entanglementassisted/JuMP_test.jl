
relaxed_tols = (default_tol_relax = 100,)
insts = OrderedDict()
insts["minimal"] = [
    ((2, 2), nothing, relaxed_tols),
    ]
insts["various"] = [
    ((2, 2), nothing, relaxed_tols),
    ((1, 4), nothing, relaxed_tols),
    ((4, 1), nothing, relaxed_tols),
    ]
return (EntanglementAssisted, insts)
