
relaxed_tols = (default_tol_relax = 100,)
insts = OrderedDict()
insts["minimal"] = [
    ((2, 3),),
    ((2, 3), :SOCExpPSD),
    ]
insts["fast"] = [
    ((5, 10), nothing, relaxed_tols),
    ((5, 10), :SOCExpPSD, relaxed_tols),
    ((10, 20), nothing, relaxed_tols),
    ((10, 20), :SOCExpPSD, relaxed_tols),
    ((20, 40), nothing, relaxed_tols),
    ((20, 40), :SOCExpPSD, relaxed_tols),
    ((40, 80), nothing, relaxed_tols),
    ((40, 80), :SOCExpPSD, relaxed_tols),
    ((100, 150), nothing, relaxed_tols),
    ((100, 150), :SOCExpPSD, relaxed_tols),
    ]
insts["various"] = [
    ((20, 40), nothing, relaxed_tols),
    ((20, 40), :SOCExpPSD, relaxed_tols),
    ((40, 80), nothing, relaxed_tols),
    ((40, 80), :SOCExpPSD, relaxed_tols),
    ((100, 300), nothing, relaxed_tols),
    ((100, 300), :SOCExpPSD, relaxed_tols),
    ]
return (RobustGeomProgJuMP, insts)
