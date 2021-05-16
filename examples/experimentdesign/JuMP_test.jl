
insts = Dict()
insts["minimal"] = [
    ((2, Cones.InvSSF()),),
    ((3, Cones.NegEntropySSF()),),
    ]
insts["fast"] = [
    ((10, Cones.InvSSF()),),
    ((15, Cones.NegLogSSF()),),
    ((20, Cones.NegEntropySSF()),),
    ((25, Cones.Power12SSF(1.5)),),
    ]
insts["various"] = [
    ((100, Cones.InvSSF()),),
    ((100, Cones.NegLogSSF()),),
    ((100, Cones.NegEntropySSF()),),
    ((100, Cones.Power12SSF(1.5)),),
    ((200, Cones.InvSSF()),),
    ((200, Cones.NegLogSSF()),),
    ((200, Cones.NegEntropySSF()),),
    ((200, Cones.Power12SSF(1.5)),),
    ((300, Cones.InvSSF()),),
    ((300, Cones.NegLogSSF()),),
    ((300, Cones.NegEntropySSF()),),
    ((300, Cones.Power12SSF(1.5)),),
    ]
return (ExperimentDesignJuMP, insts)
