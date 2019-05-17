#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

import JLD
import Random

examples_dir = joinpath(@__DIR__, "../examples")

include(joinpath(examples_dir, "densityest/JuMP.jl"))
include(joinpath(examples_dir, "envelope/native.jl"))
include(joinpath(examples_dir, "expdesign/JuMP.jl"))
include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "lotkavolterra/JuMP.jl"))
include(joinpath(examples_dir, "polymin/native.jl"))

# TODO add families of instances for each model

outputpath = joinpath(@__DIR__, "instancefiles", "jld")

Random.seed!(1234)

function example_to_JLD(modelname::String, isnative::Bool)
    d = eval(Meta.parse(modelname))()
    if isnative
        (c, A, b, G, h, cones, cone_idxs) = (d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    else
        model = d.model
        model.moi_backend.optimizer.model.optimizer.load_only = true
        JuMP.optimize!(model)
        opt = model.moi_backend.optimizer.model.optimizer
        (c, A, b, G, h, cones, cone_idxs) = (opt.c, opt.A, opt.b, opt.G, opt.h, opt.cones, opt.cone_idxs)
    end
    fullpathout = joinpath(outputpath, modelname * ".jld")
    JLD.save(fullpathout, "c", c, "A", A, "b", b, "G", G, "h", h, "cones", cones, "cone_idxs", cone_idxs)
    return
end

for (modelname, isnative) in [
    ("densityest", false),
    ("envelope1", true),
    ]
    example_to_JLD(modelname, isnative)
end
