
insts = OrderedDict()
insts["minimal"] = [
    ((3, :NegLogSSF, false, false),),
    ((3, :NegLogSSF, true, false),),
    ((3, :Power12SSF, false, false),),
    ((3, :Power12SSF, true, false),),
    ((3, :InvSSF, true, true),),
    ]
insts["fast"] = [
    ((10, :InvSSF),),
    ((15, :NegLogSSF),),
    ((20, :NegEntropySSF),),
    ((25, :Power12SSF),),
    ]
insts["various"] = [
    ((20, :InvSSF),),
    ((20, :NegLogSSF),),
    ((20, :NegEntropySSF),),
    ((20, :Power12SSF),),
    ((50, :InvSSF),),
    ((50, :NegLogSSF),),
    ((50, :NegEntropySSF),),
    ((50, :Power12SSF),),
    ]
return (CovarianceEstJuMP, insts)
