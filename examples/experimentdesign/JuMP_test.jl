
insts = OrderedDict()
insts["minimal"] = [
    # rootdet
    ((6, MatNegGeom()),),
    ((6, MatNegGeomEFExp()),),
    ((6, MatNegGeomEFPow()),),
    # tr inv
    ((4, MatInv()),),
    ((4, MatInvEigOrd()),),
    ((4, MatInvDirect()),),
    # tr neglog
    ((4, MatNegLog()),),
    ((4, MatNegLogEigOrd()),),
    ((4, MatNegLogDirect()),),
    # tr negentropy
    ((4, MatNegEntropy()),),
    ((4, MatNegEntropyEigOrd()),),
    # tr power12
    ((5, MatPower12(1.5)),),
    ((5, MatPower12EigOrd(1.5)),),
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
