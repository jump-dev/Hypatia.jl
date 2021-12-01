
function experimentdesign_insts(ext::MatSpecExt)
    return [[(d, ext) for d in vcat(3, 25:25:100, 150, 200:100:900)]]
end

insts = OrderedDict()
insts["nat"] = (
    nothing,
    vcat(
        experimentdesign_insts.([
            MatNegRtdet(),
            MatNegLog(),
            MatNegSqrtConj(),
            MatNegPower01(1 / 3),
        ])...,
    ),
)
insts["natlog"] = (nothing, experimentdesign_insts(MatLogdetCone()))
insts["ext"] = (
    nothing,
    vcat(
        experimentdesign_insts.([
            MatNegRtdetEFExp(),
            MatNegLogDirect(),
            MatNegSqrtConjDirect(),
            MatNegPower01EigOrd(1 / 3),
        ])...,
    ),
)
return (ExperimentDesignJuMP, insts)
