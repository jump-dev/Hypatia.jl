#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

TODO readme for benchmarks and describe ARGS for running on command line
=#

println("must run from Hypatia/benchmark directory") # TODO delete later

using Hypatia
using MathOptFormat
using Dates


# parse command line arguments
println()
if length(ARGS) != 3
    error("usage: julia runbenchmarks.jl instance_set input_path output_path")
end

instanceset = ARGS[1]
instsetfile = joinpath(@__DIR__, "instancesets", instanceset * ".txt")
if !isfile(instsetfile)
    error("instance set file not found: $instsetfile")
end

inputpath = ARGS[2]
if !isdir(inputpath)
    error("input path is not a valid directory: $inputpath")
end

# check that each instance is in the inputpath
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
println("instance set $instanceset contains $(length(instances)) instances")
for instname in instances
    instfile = joinpath(inputpath, instname)
    if !isfile(instfile)
        error("instance file not found: $instfile")
    end
end

outputpath = ARGS[3]
if !isdir(outputpath)
    error("output path is not a valid directory: $outputpath")
end


# TODO these options
# timelimit = ARGS[4]
lscachetype = "QRSymmCache" # linear system solver cache type
if !in(lscachetype, ("QRSymmCache", "NaiveCache"))
    error("linear system cache type $lscachetype is not recognized")
end
usedense = parse(Bool, "false") # whether A and G matrices are represented as dense or sparse

println("\nlinear systems using $lscachetype")
println("matrices A, G will be $(usedense ? "dense" : "sparse")")

# Hypatia options
options = Dict()
options[:verbose] = true
options[:timelimit] = 1.8e3
options[:maxiter] = 1000

println("Hypatia options are:")
for (k, v) in options
    println("  $k = $v")
end


using MathOptInterface
MOI = MathOptInterface
MOIT = MOI.Test
MOIB = MOI.Bridges
MOIU = MOI.Utilities

MOIU.@model(HypatiaModelData,
    (),
    (
        MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval,
    ),
    (
        MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
        MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
        MOI.ExponentialCone, MOI.PowerCone, MOI.GeometricMeanCone,
        MOI.PositiveSemidefiniteConeTriangle,
        MOI.LogDetConeTriangle,
    ),
    (),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    )




println("\nstarting benchmark run in 5 seconds\n")
sleep(5.0)

# each line of csv file will summarize Hypatia performance on a particular instance
csvfile = joinpath(outputpath, "RESULTS_$(instanceset).csv")
open(csvfile, "w") do fdcsv
    println(fdcsv, "instname,status,pobj,dobj,niters,runtime,gctime,bytes")
end

# run each instance, print Hypatia output to instance-specific file, and print results to a single csv file
OUT = stdout
ERR = stderr
for instname in instances
    println("starting $instname")

    solveerror = nothing
    (status, pobj, dobj, niters, runtime, gctime, bytes) = (:UnSolved, NaN, NaN, -1, NaN, NaN, -1)
    memallocs = nothing

    instfile = joinpath(outputpath, instname * ".txt")
    open(instfile, "w") do fdinst
        redirect_stdout(fdinst)
        redirect_stderr(fdinst)

        println("instance $instname")
        println("ran at: ", Dates.now())
        println()
        println("linear systems:  $lscachetype")
        println("matrices A, G:   $(usedense ? "dense" : "sparse")")
        println("Hypatia options:")
        for (k, v) in options
            println("  $k = $v")
        end

        println("\nreading instance and constructing model...")
        optimizer = MOIU.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.Optimizer(
            verbose = verbose,
            timelimit = 2e1,
            lscachetype = lscachetype,
            usedense = usedense,
            tolrelopt = 1e-6,
            tolabsopt = 1e-7,
            tolfeas = 1e-7,
            ))
        readtime = @elapsed MOI.read_from_file(optimizer, joinpath(inputpath, instname))
        println("took $readtime seconds")
        # if hasintvars
        #     println("ignoring integrality constraints")
        # end

        println("\nsolving model...")
        try
            (val, runtime, bytes, gctime, memallocs) = @timed MOI.optimize!(optimizer)
            println("\nHypatia finished")
            status = MOI.get(optimizer, MOI.TerminationStatus())
            # niters = MOI.get(optimizer, MOI.BarrierIterations())
            niters = -1 # TODO
            pobj = MOI.get(optimizer, MOI.ObjectiveValue())
            dobj = MOI.get(optimizer, MOI.ObjectiveBound())
        catch solveerror
            println("\nHypatia errored:")
            println(solveerror)
        end
        println("took $runtime seconds")
        println("memory allocation data:")
        dump(memallocs)
        println()

        redirect_stdout(OUT)
        redirect_stderr(ERR)
    end

    if !isnothing(solveerror)
        println("Hypatia errored:")
        println(solveerror)
        println()
    end

    open(csvfile, "a") do fdcsv
        println(fdcsv, "$instname,$status,$pobj,$dobj,$niters,$runtime,$gctime,$bytes")
    end
end
