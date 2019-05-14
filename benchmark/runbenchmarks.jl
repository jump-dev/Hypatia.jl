#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

TODO readme for benchmarks and describe ARGS for running on command line
=#

import Hypatia
import MathOptFormat
import MathOptInterface
const MOI = MathOptInterface
import GZip
import Dates

# parse command line arguments
println()
if length(ARGS) != 3
    error("usage: julia runbenchmarks.jl instance_set input_path output_path")
end

instanceset = ARGS[1]
instsetfile = joinpath(@__DIR__, "instancesets", instanceset)
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

# Hypatia options
verbose = true
time_limit = 1e2
use_dense = false

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

optimizer = MOI.Utilities.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.Optimizer(
    verbose = verbose,
    time_limit = time_limit,
    use_dense = use_dense,
    tol_rel_opt = 1e-6,
    tol_abs_opt = 1e-7,
    tol_feas = 1e-7,
    ))

println("\nstarting benchmark run in 5 seconds\n")
sleep(5.0)

# each line of csv file will summarize Hypatia performance on a particular instance
csvfile = joinpath(outputpath, "RESULTS_$(instanceset).csv")
open(csvfile, "w") do fdcsv
    println(fdcsv, "instname,status,primal_obj,dual_obj,niters,runtime,gctime,bytes")
end

# run each instance, print Hypatia output to instance-specific file, and print results to a single csv file
OUT = stdout
ERR = stderr
for instname in instances
    println("starting $instname")

    solveerror = nothing
    (status, primal_obj, dual_obj, niters, runtime, gctime, bytes) = (:UnSolved, NaN, NaN, -1, NaN, NaN, -1)
    memallocs = nothing

    instfile = joinpath(outputpath, instname * ".txt")
    open(instfile, "w") do fdinst
        redirect_stdout(fdinst)
        redirect_stderr(fdinst)

        println("instance $instname")
        println("ran at: ", Dates.now())
        println()

        println("\nreading instance and constructing model...")
        readtime = @elapsed begin
            model = MathOptFormat.read_into_model(joinpath(inputpath, instname))
            MOI.empty!(optimizer)
            MOI.copy_to(optimizer, model)
        end
        println("took $readtime seconds")

        println("\nsolving model...")
        try
            (val, runtime, bytes, gctime, memallocs) = @timed MOI.optimize!(optimizer)
            println("\nHypatia finished")
            status = MOI.get(optimizer, MOI.TerminationStatus())
            niters = -1 # TODO niters = MOI.get(optimizer, MOI.BarrierIterations())
            primal_obj = MOI.get(optimizer, MOI.ObjectiveValue())
            dual_obj = MOI.get(optimizer, MOI.ObjectiveBound())
        catch solveerror
            println("\nHypatia errored: ", solveerror)
        end
        println("took $runtime seconds")
        println("memory allocation data:")
        dump(memallocs)
        println()

        redirect_stdout(OUT)
        redirect_stderr(ERR)
    end

    if !isnothing(solveerror)
        println("Hypatia errored: ", solveerror)
    end

    open(csvfile, "a") do fdcsv
        println(fdcsv, "$instname,$status,$primal_obj,$dual_obj,$niters,$runtime,$gctime,$bytes")
    end
end

println("\ndone\n")
