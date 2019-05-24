#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

import JLD
import Random
import JuMP
import Hypatia

examples_dir = joinpath(@__DIR__, "../examples")
for e in readdir(examples_dir)
    e_dir = joinpath(examples_dir, e)
    if !isdir(e_dir)
        continue
    end
    for ef in readdir(e_dir)
        if ef == "JuMP.jl" || ef == "native.jl"
            include(joinpath(e_dir, ef))
        end
    end
end

outputpath = joinpath(@__DIR__, "instancefiles", "jld")

Random.seed!(1234)

function example_to_JLD(modelname::String, isnative::Bool)
    d = eval(Meta.parse(modelname))()
    if isnative
        (c, A, b, G, h, cones, cone_idxs) = (d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    else
        model = d.model
        JuMP.optimize!(model, JuMP.with_optimizer(Hypatia.Optimizer, load_only = true))
        opt = model.moi_backend.optimizer.model.optimizer
        (c, A, b, G, h, cones, cone_idxs) = (opt.c, opt.A, opt.b, opt.G, opt.h, opt.cones, opt.cone_idxs)
    end
    fullpathout = joinpath(outputpath, modelname * ".jld")
    JLD.save(fullpathout, "c", c, "A", A, "b", b, "G", G, "h", h, "cones", cones, "cone_idxs", cone_idxs)
    return
end

for (modelname, isnative) in [
    ("densityestJuMP5", false),
    ("envelope1", true),
    ]
    example_to_JLD(modelname, isnative)
end
