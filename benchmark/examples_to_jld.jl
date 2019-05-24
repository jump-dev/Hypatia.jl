#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

import JLD
import Random
import JuMP
import Hypatia
import Random

# instanceset = ARGS[1]
instanceset = "JuMP_easy"
instsetfile = joinpath(@__DIR__, "instancesets", instanceset * ".txt")
isnative = false # TODO infer
outputpath = joinpath(@__DIR__, "instancefiles")
if !isdir(outputpath)
    error("output path is not a valid directory: $outputpath")
end

Random.seed!(1234) # would make more sense for random seed to be in all the model functions, not outside

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

function example_to_JLD(modelname::AbstractString, isnative::Bool)
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

for l in readlines(instsetfile)
    str = split(strip(l))
    if !isempty(str)
        str1 = first(str)
        if !startswith(str1, '#')
            println("\nconverting $(str1)\n")
            example_to_JLD(str1, isnative)
        end
    end
end
