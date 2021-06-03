
insts = OrderedDict()
insts["minimal"] = [
    # tr neglog
    ((3, MatNegLog()),),
    ((3, MatNegLogEigOrd()),),
    ((3, MatNegLogDirect()),),
    # tr negentropy
    ((3, MatNegEntropy()),),
    ((3, MatNegEntropyEigOrd()),),
    # tr power12
    ((3, MatPower12(1.5)),),
    ((3, MatPower12EigOrd(1.5)),),
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
