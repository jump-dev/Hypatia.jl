
insts = OrderedDict()
insts["minimal"] = [
    ((2, :InvSSF, false),),
    ((2, :InvSSF, true),),
    ((3, :NegEntropySSF, false),),
    ((3, :NegEntropySSF, true),),
    ]
insts["fast"] = [
    ((10, :InvSSF),),
    ((15, :NegLogSSF),),
    ((20, :NegEntropySSF),),
    ((25, :Power12SSF),),
    ]
insts["various"] = [
    ((100, :InvSSF),),
    ((100, :NegLogSSF),),
    ((100, :NegEntropySSF),),
    ((100, :Power12SSF),),
    ((200, :InvSSF),),
    ((200, :NegLogSSF),),
    ((200, :NegEntropySSF),),
    ((200, :Power12SSF),),
    ((300, :InvSSF),),
    ((300, :NegLogSSF),),
    ((300, :NegEntropySSF),),
    ((300, :Power12SSF),),
    ]
return (ExperimentDesignJuMP, insts)
