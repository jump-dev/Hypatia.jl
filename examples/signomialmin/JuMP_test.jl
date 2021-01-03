
insts = Dict()
insts["minimal"] = [
    ((2, 2),),
    ((2, 2), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    # ((:motzkin2,),),
    # ((:motzkin2,), SOCExpPSDOptimizer),
    # ((:motzkin3,),),
    # ((:CS16ex8_13,),),
    # ((:CS16ex8_14,),),
    # ((:CS16ex18,),),
    # ((:CS16ex12,),),
    # ((:CS16ex13,),),
    # ((:MCW19ex1_mod,),),
    # ((:MCW19ex8,),),
    # ((:MCW19ex8,), SOCExpPSDOptimizer),
    # ((3, 2),),
    # ((3, 2), SOCExpPSDOptimizer),
    # ((6, 6),),
    ((20, 3),),
    # ((20, 3), SOCExpPSDOptimizer),
    ]
insts["slow"] = [
    ((10, 10),),
    ((10, 10), SOCExpPSDOptimizer),
    ((20, 6),),
    ((40, 3),),
    ]
return (SignomialMinJuMP, insts)
