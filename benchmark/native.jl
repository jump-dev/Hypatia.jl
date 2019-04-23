#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using JLD2, FileIO
import JuMP
import Hypatia
const HYP = Hypatia
const MO = HYP.Models

examples_dir = joinpath(@__DIR__, "../examples")

include(joinpath(examples_dir, "envelope/native.jl"))
# include(joinpath(examples_dir, "linearopt/native.jl"))
# include(joinpath(examples_dir, "polymin/real.jl"))
# include(joinpath(examples_dir, "polymin/complex.jl"))
# include(joinpath(examples_dir, "envelope/jump.jl"))
include(joinpath(examples_dir, "expdesign/jump.jl"))
# include(joinpath(examples_dir, "polymin/jump.jl"))
# include(joinpath(examples_dir, "shapeconregr/jump.jl"))
# include(joinpath(examples_dir, "densityest/jump.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmatrix.jl"))
# include(joinpath(examples_dir, "wsosmatrix/muconvexity.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmat1.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmat2.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmat3.jl"))
# include(joinpath(examples_dir, "regionofattraction/univariate.jl"))
# include(joinpath(examples_dir, "contractionanalysis/jump.jl"))

instancedir = joinpath(@__DIR__, "instancefiles", "jld")

function make_JLD(modelname::String)
    outputpath = joinpath(instancedir, modelname)
    if modelname == "envelope.jld2"
        (c, A, b, G, h, cones, cone_idxs) =  build_envelope(3, 5, 3, 5, primal_wsos = true, dense = true)
    elseif modelname == "expdesign"
        (q, p, n, nmax) = (5, 15, 25, 5)
        V = randn(q, p)
        (model, _) = build_JuMP_expdesign(q, p, V, n, nmax)
        # set up optimizer...
        nativedata = model.moi_backend.optimizer.model.optimizer
        (c, A, b, G, h, cones, cone_idxs) = (nativedata.c, nativedata.A, nativedata.b, nativedata.G, nativedata.h, nativedata.cones, nativedata.cone_idxs)
    end
    @save outputpath c A b G h cones cone_idxs
    return nothing
end


modelname = "envelope.jld2"
function make_model(modelname::String)
    load(joinpath(instancedir, modelname), "c", "A", "b", "G", "h", "cones", "cone_idxs")
    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    # solver = SO.HSDSolver(model, verbose = true)
    # SO.solve(solver)
    return model
end
