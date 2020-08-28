
insts = Dict()
insts["minimal"] = [
    ((2, 2),),
    ((2, 2), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((:motzkin2,),),
    ((:motzkin2,), StandardConeOptimizer),
    ((:motzkin3,),),
    ((:CS16ex8_13,),),
    ((:CS16ex8_14,),),
    ((:CS16ex18,),),
    ((:CS16ex12,),),
    ((:CS16ex13,),),
    ((:MCW19ex1_mod,),),
    ((:MCW19ex8,),),
    ((:MCW19ex8,), StandardConeOptimizer),
    ((3, 2),),
    ((3, 2), StandardConeOptimizer),
    ((6, 6),),
    ((20, 3),),
    ((20, 3), StandardConeOptimizer),
    ]
insts["slow"] = [
    ((10, 10),),
    ((10, 10), StandardConeOptimizer),
    ((20, 6),),
    ((40, 3),),
    ]
return (SignomialMinJuMP, insts)
