#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

# julia benchmark/cbf_to_jld.jl cbf_easy C:/Users/lkape/Documents/cblib/all/cblib.zib.de/download/all benchmark/instancefiles
Pkg.activate(".")

import JLD
import MathOptFormat
import MathOptInterface
const MOI = MathOptInterface
import Hypatia

instanceset = ARGS[1]
instsetfile = joinpath(@__DIR__, "instancesets", instanceset * ".txt")

inputpath = ARGS[2]

outputpath = ARGS[3]
if !isdir(outputpath)
    error("output path is not a valid directory: $outputpath")
end

instances = SubString[]
for l in readlines(instsetfile)
    str = split(strip(l))
    if !isempty(str)
        str1 = first(str)
        if !startswith(str1, '#')
            push!(instances, str1)
        end
    end
end

MOI.Utilities.@model(HypatiaModelData,
    (MOI.Integer,), # integer constraints will be ignored by Hypatia
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
        MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
        MOI.PositiveSemidefiniteConeTriangle,
        MOI.ExponentialCone),
    (MOI.PowerCone,),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    )

optimizer = MOI.Utilities.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.Optimizer())

for instname in instances
    println("opening $instname")
    fullpathin = joinpath(inputpath, instname * ".cbf.gz")
    model = MathOptFormat.read_from_file(fullpathin)
    MOI.empty!(optimizer)
    MOI.copy_to(optimizer, model)
    d = optimizer.optimizer
    # just load, don't optimize
    d.use_dense = false
    d.load_only = true
    MOI.optimize!(optimizer)
    (c, A, b, G, h, cones, cone_idxs) = (d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    fullpathout = joinpath(outputpath, instname * ".jld")
    JLD.save(fullpathout, "c", c, "A", A, "b", b, "G", G, "h", h, "cones", cones, "cone_idxs", cone_idxs)
end


;
