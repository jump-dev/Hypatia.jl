
insts = Dict()
insts["minimal"] = [
    ((2, true, false),),
    ((2, true, true),),
    ((2, false, true),),
    ((2, false, true), StandardConeOptimizer),
    ((2, false, true), SOPSDConeOptimizer),
    # ((2, false, true), ExpConeOptimizer), # TODO waiting for MOI bridges geomean to exp
    ]
insts["fast"] = [
    ((10, true, false),),
    ((10, false, true),),
    ((10, false, true), StandardConeOptimizer),
    ((10, true, true),),
    ((100, true, false),),
    ((100, false, true),),
    ((100, false, true), StandardConeOptimizer),
    ((100, true, true),),
    ((1000, true, false),),
    ((1000, true, true),), # with bridges extended formulation will need to go into slow list
    ]
insts["slow"] = [
    ((1000, false, true), StandardConeOptimizer),
    ((2000, true, false),),
    ((2000, false, true),),
    ((2000, true, true),),
    ]
return (MaxVolumeJuMP, insts)
