
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
insts["slow"] = []
insts["various"] = [
    ((20, Cones.InvSSF()),),
    ((20, Cones.NegLogSSF()),),
    ((20, Cones.NegEntropySSF()),),
    ((20, Cones.Power12SSF(1.5)),),
    ((50, Cones.InvSSF()),),
    ((50, Cones.NegLogSSF()),),
    ((50, Cones.NegEntropySSF()),),
    ((50, Cones.Power12SSF(1.5)),),
    ]
return (CovarianceEstJuMP, insts)
