#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

# import JLD2
import JLD
# import FileIO
import JuMP
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models # TODO remove, for loading
const SO = HYP.Solvers # TODO remove, for loading
import SparseArrays # TODO remove, for loading
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
include(joinpath(examples_dir, "regionofattraction/univariate.jl"))
# include(joinpath(examples_dir, "contractionanalysis/jump.jl"))

# TODO account for use_dense
# TODO add families of instances for each model

outputpath = joinpath(@__DIR__, "instancefiles", "jld")

Random.seed!(1234)

function make_JLD(modelname::String)
    if modelname == "envelope"
        (c, A, b, G, h, cones, cone_idxs) = build_envelope(3, 5, 3, 5, primal_wsos = true, dense = true)
    elseif modelname == "linearopt"
        (c, A, b, G, h, cones, cone_idxs) = build_linearopt(15, 20)
    elseif modelname == "polyminreal1"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:butcher, 2)
    elseif modelname == "polyminreal2"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:caprasse, 4)
    elseif modelname == "polyminreal3"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:goldsteinprice, 7)
    elseif modelname == "polyminreal4"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:heart, 2)
    elseif modelname == "polyminreal5"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:lotkavolterra, 3)
    elseif modelname == "polyminreal6"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:magnetism7, 2)
    elseif modelname == "polyminreal7"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:motzkin, 7)
    elseif modelname == "polyminreal8"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:reactiondiffusion, 4)
    elseif modelname == "polyminreal9"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:robinson, 8)
    elseif modelname == "polyminreal10"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:rosenbrock, 4)
    elseif modelname == "polyminreal11"
        (c, A, b, G, h, cones, cone_idxs) = build_polymin(:schwefel, 3)
    elseif modelname == "polymincomplex1"
        (n, deg, f, gs, gdegs, _) = complexpolys[:abs1d]
        d = deg # no less than
        (c, A, b, G, h, cones, cone_idxs) = build_complexpolymin(n, d, f, gs, gdegs, false)
    elseif modelname == "polymincomplex2"
        (n, deg, f, gs, gdegs, _) = complexpolys[:absunit1d]
        d = deg # no less than
        (c, A, b, G, h, cones, cone_idxs) = build_complexpolymin(n, d, f, gs, gdegs, false)
    elseif modelname == "polymincomplex3"
        (n, deg, f, gs, gdegs, _) = complexpolys[:negabsunit1d]
        d = deg # no less than
        (c, A, b, G, h, cones, cone_idxs) = build_complexpolymin(n, d, f, gs, gdegs, false)
    elseif modelname == "polymincomplex4"
        (n, deg, f, gs, gdegs, _) = complexpolys[:absball2d]
        d = deg # no less than
        (c, A, b, G, h, cones, cone_idxs) = build_complexpolymin(n, d, f, gs, gdegs, false)
    elseif modelname == "polymincomplex5"
        (n, deg, f, gs, gdegs, _) = complexpolys[:absbox2d]
        d = deg # no less than
        (c, A, b, G, h, cones, cone_idxs) = build_complexpolymin(n, d, f, gs, gdegs, false)
    elseif modelname == "polymincomplex6"
        (n, deg, f, gs, gdegs, _) = complexpolys[:negabsbox2d]
        d = deg # no less than
        (c, A, b, G, h, cones, cone_idxs) = build_complexpolymin(n, d, f, gs, gdegs, false)
    elseif modelname == "polymincomplex7"
        (n, deg, f, gs, gdegs, _) = complexpolys[:denseunit1d]
        d = deg # no less than
        (c, A, b, G, h, cones, cone_idxs) = build_complexpolymin(n, d, f, gs, gdegs, false)
    else
        if modelname == "expdesign_small"
            (q, p, n, nmax) = (5, 15, 25, 5)
            V = randn(q, p)
            (model, _) = build_JuMP_expdesign(q, p, V, n, nmax)
        elseif modelname == "expdesign_medium"
            (q, p, n, nmax) = (10, 30, 50, 5)
            V = randn(q, p)
            (model, _) = build_JuMP_expdesign(q, p, V, n, nmax)
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
        elseif modelname == "roa"
            deg = 4
            model = univariate_WSOS(deg)
        else
            error("unknown model name")
        end
        model.moi_backend.optimizer.model.optimizer.load_only = true
        JuMP.optimize!(model)
        nativedata = model.moi_backend.optimizer.model.optimizer
        (c, A, b, G, h, cones, cone_idxs) = (nativedata.c, nativedata.A, nativedata.b, nativedata.G, nativedata.h, nativedata.cones, nativedata.cone_idxs)
    end
    fullpathout = joinpath(outputpath, modelname * ".jld")
    # JLD.save(fullpathout, "modeldata", (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs))
    JLD.save(fullpathout, "c", c, "A", A, "b", b, "G", G, "h", h, "cones", cones, "cone_idxs", cone_idxs)
    return nothing
end

for modelname in [
    "envelope",
    "linearopt",
    "polyminreal1",
    "polyminreal2",
    "polyminreal3",
    "polyminreal4",
    "polyminreal5",
    "polyminreal6",
    "polyminreal7",
    "polyminreal8",
    "polyminreal9",
    "polyminreal10",
    "polyminreal11",
    "polymincomplex1",
    "polymincomplex2",
    "polymincomplex3",
    "polymincomplex4",
    "polymincomplex5",
    "polymincomplex6",
    "polymincomplex7",
    "expdesign_small",
    "expdesign_medium",
    "shapeconregr",
    "densityest",
    "roa",
    ]
    make_JLD(modelname)
end


modelname = "envelope"
function make_model(modelname::String)
    # md = FileIO.load(joinpath(outputpath, modelname * ".jld2"), "modeldata")
    md = JLD.load(joinpath(outputpath, modelname * ".jld"))
    (c, A, b, G, h, cones, cone_idxs) = (md["c"], md["A"], md["b"], md["G"], md["h"], md["cones"], md["cone_idxs"])
    for c in cones
        CO.setup_data(c)
    end
    # plmodel = MO.PreprocessedLinearModel(md.c, md.A, md.b, md.G, md.h, md.cones, md.cone_idxs)
    plmodel = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(plmodel, verbose = true)
    SO.solve(solver)
    return nothing
end

# for modelname in [
#     "envelope",
#     "linearopt",
#     "polyminreal",
#     "polymincomplex",
#     "expdesign",
#     "shapeconregr",
#     "densityest",
#     ]
#     make_model(modelname)
# end
