#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using JLD2, FileIO
import JuMP
import Hypatia
# import MathOptInterface
# const MOI = MathOptInterface
# const MOIU = MOI.Utilities
const HYP = Hypatia
const MO = HYP.Models
const SO = HYP.Solvers
import Random
import Distributions

examples_dir = joinpath(@__DIR__, "../examples")

include(joinpath(examples_dir, "envelope/native.jl"))
include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "polymin/real.jl"))
include(joinpath(examples_dir, "polymin/complex.jl"))
# include(joinpath(examples_dir, "envelope/jump.jl"))
include(joinpath(examples_dir, "expdesign/jump.jl"))
# include(joinpath(examples_dir, "polymin/jump.jl"))
include(joinpath(examples_dir, "shapeconregr/jump.jl"))
include(joinpath(examples_dir, "densityest/jump.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmatrix.jl"))
# include(joinpath(examples_dir, "wsosmatrix/muconvexity.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmat1.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmat2.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmat3.jl"))
# include(joinpath(examples_dir, "regionofattraction/univariate.jl"))
# include(joinpath(examples_dir, "contractionanalysis/jump.jl"))

# TODO account for use_dense
# TODO add families of instances for each model

instancedir = joinpath(@__DIR__, "instancefiles", "jld")

Random.seed!(1234)

function make_JLD(modelname::String)
    outputpath = joinpath(instancedir, modelname)
    if modelname == "envelope"
        (c, A, b, G, h, cones, cone_idxs) = build_envelope(3, 5, 3, 5, primal_wsos = true, dense = true)
    elseif modelname == "linearopt"
        (c, A, b, G, h, cones, cone_idxs) = build_linearopt(15, 20)
    elseif modelname == "polyminreal"
        (c, A, b, G, h, cones, cone_idxs) =  build_polymin(:reactiondiffusion, 4)
    elseif modelname == "polymincomplex"
        (n, deg, f, gs, gdegs, _) = complexpolys[:negabsbox2d]
        d = deg # no less than
        (c, A, b, G, h, cones, cone_idxs) = build_complexpolymin(n, d, f, gs, gdegs, false)
    else
        if modelname == "expdesign"
            (q, p, n, nmax) = (5, 15, 25, 5)
            V = randn(q, p)
            (model, _) = build_JuMP_expdesign(q, p, V, n, nmax)
            # load_only_optimizer = JuMP.with_optimizer(HYP.Optimizer, verbose = true, load_only = true)
            # JuMP.set_optimizer(model, load_only_optimizer)
        elseif modelname == "shapeconregr"
            (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^3))
            shape_data = ShapeData(n)
            (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 3.0)
            model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, use_dense = true))
            (coeffs, polys) = build_shapeconregr_WSOS(model, X, y, deg, shape_data)
        elseif modelname == "densityest"
            nobs = 200; n = 1; deg = 4; X = rand(Distributions.Uniform(-1, 1), nobs, n); dom = MU.Box(-ones(n), ones(n))
            model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer))
            (interp_bases, coeffs) = build_JuMP_densityest(model, X, deg, dom, use_monomials = false)
        else
            error("unknown model name")
        end
        model.moi_backend.optimizer.model.optimizer.load_only = true
        JuMP.optimize!(model)
        nativedata = model.moi_backend.optimizer.model.optimizer
        (c, A, b, G, h, cones, cone_idxs) = (nativedata.c, nativedata.A, nativedata.b, nativedata.G, nativedata.h, nativedata.cones, nativedata.cone_idxs)
    end
    @save outputpath * ".jld2" c A b G h cones cone_idxs
    return nothing
end

for modelname in [
    "envelope",
    "linearopt",
    "polyminreal",
    "polymincomplex",
    "expdesign",
    "shapeconregr",
    "densityest",
    ]
    make_JLD(modelname)
end


modelname = "densityest"
function make_model(modelname::String)
    load(joinpath(instancedir, modelname * ".jld2"), "c", "A", "b", "G", "h", "cones", "cone_idxs")
    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = true)
    SO.solve(solver)
    return model
end

# for modelname in [
#     "envelope",
#     "expdesign",
#     ]
#     make_model(modelname)
# end
