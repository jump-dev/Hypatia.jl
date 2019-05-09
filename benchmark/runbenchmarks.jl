#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

TODO readme for benchmarks and describe ARGS for running on command line
=#

# julia benchmark/runbenchmarks.jl native_all.txt benchmark/instancefiles/jld tmp

Pkg.activate(".")
import Hypatia
const CO = Hypatia.Cones
const MO = Hypatia.Models
const SO = Hypatia.Solvers
import JLD
import Dates
import SparseArrays
import LinearAlgebra

# parse command line arguments
println()
if length(ARGS) != 3
    error("usage: julia runbenchmarks.jl instance_set input_path output_path")
end

instanceset = ARGS[1]
# instanceset = "native_all.txt"
instsetfile = joinpath(@__DIR__, "instancesets", "jld", instanceset)
if !isfile(instsetfile)
    error("instance set file not found: $instsetfile")
end

inputpath = ARGS[2]
# inputpath = joinpath(@__DIR__, "instancefiles", "jld")
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
# outputpath = "tmp"
if !isdir(outputpath)
    error("output path is not a valid directory: $outputpath")
end

# Hypatia options
verbose = true
time_limit = 1e2

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

    instfile = joinpath(outputpath, chop(instname, tail = 4) * ".txt")
    open(instfile, "w") do fdinst
        redirect_stdout(fdinst)
        redirect_stderr(fdinst)

        println("instance $instname")
        println("ran at: ", Dates.now())
        println()

        println("\nreading instance and constructing model...")
        readtime = @elapsed begin
            md = JLD.load(joinpath(inputpath, instname))
            (c, A, b, G, h, cones, cone_idxs) = (md["c"], md["A"], md["b"], md["G"], md["h"], md["cones"], md["cone_idxs"])
            for c in cones
                CO.setup_data(c)
            end
            plmodel = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
            solver = SO.HSDSolver(plmodel, verbose = verbose, time_limit = time_limit)
        end
        println("took $readtime seconds")

        println("\nsolving model...")
        try
            (val, runtime, bytes, gctime, memallocs) = @timed SO.solve(solver)
            println("\nHypatia finished")
            status = solver.status
            niters = solver.num_iters
            primal_obj = solver.primal_obj
            dual_obj = solver.dual_obj
        catch solveerror
            println("\nHypatia errored: ", solveerror)
        end
        println("took $runtime seconds")
        println("memory allocation data:")
        dump(memallocs)
        println()

        # redirect_stdout(OUT)
        # redirect_stderr(ERR)
    end

    if !isnothing(solveerror)
        println("Hypatia errored: ", solveerror)
    end

    open(csvfile, "a") do fdcsv
        println(fdcsv, "$instname,$status,$primal_obj,$dual_obj,$niters,$runtime,$gctime,$bytes")
    end
end

println("\ndone\n")
