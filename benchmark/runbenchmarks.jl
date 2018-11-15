#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

TODO readme file for benchmarks and describe ARGS for running on command line
=#

println("must run from Hypatia/benchmark directory") # TODO delete later

using Pkg; Pkg.activate("..") # TODO delete later
using Hypatia

# module containing functions for translating from cbf to Hypatia native format
# TODO replace with ConicBenchmarkUtilities -> MOI -> Hypatia when CBU is updated for MOI
include(joinpath(@__DIR__, "Translate", "Translate.jl"))


# parse command line arguments
println()
if length(ARGS) != 3
    error("usage: julia runbenchmarks.jl instanceset cbfpath outputpath")
end

instanceset = ARGS[1]
instsetfile = joinpath(@__DIR__, "instancesets", instanceset * ".txt")
if !isfile(instsetfile)
    error("instance set not found: $instsetfile")
end

cbfpath = ARGS[2]
if !isdir(cbfpath)
    error("cbf path is not a valid directory: $cbfpath")
end

# check that each instance is in the cbfpath
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
    instfile = joinpath(cbfpath, instname * ".cbf.gz")
    if !isfile(instfile)
        error("instance CBF file not found: $instfile")
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
# options[:maxiter] = 300
options[:verbose] = true
# options[:timelimit] = # TODO args option

println("Hypatia options are:")
for (k, v) in options
    println("  $k = $v")
end

println("\nstarting benchmark run in 5 seconds\n")
sleep(5.0)


# meta text file will contain information about the benchmarking process
metafile = joinpath(outputpath, "META_$(instanceset).txt")
# each line of csv file will summarize Hypatia performance on a particular instance
csvfile = joinpath(outputpath, "RESULTS_$(instanceset).csv")

open(metafile, "w") do fdmeta
    println(fdmeta, "instance set:    $instanceset")
    println(fdmeta, "linear systems:  $lscachetype")
    println(fdmeta, "matrices A, G:   $(usedense ? "dense" : "sparse")")
    println(fdmeta, "Hypatia options:")
    for (k, v) in options
        println(fdmeta, "  $k = $v")
    end
end

open(csvfile, "w") do fdcsv
    println(fdcsv, "instname,status,pobj,dobj,niters,runtime,gctime,bytes")
end

# run each instance, print Hypatia output to instance-specific file, and print metadata to a single csv file
OUT = stdout
ERR = stderr
for instname in instances
    println("$instname ...")

    solveerror = nothing
    (status, pobj, dobj, niters, runtime, gctime, bytes) = (:UnSolved, NaN, NaN, -1, NaN, NaN, -1)

    instfile = joinpath(outputpath, instname * ".txt")
    open(instfile, "w") do fdinst
        # redirect_stdout(fdinst)
        # redirect_stderr(fdinst)

        println("reading CBF data")
        cbfdata = Translate.readcbfdata(joinpath(cbfpath, instname * ".cbf.gz"))
        (c, A, b, G, h, cone, objoffset, hasintvars) = Translate.cbftohypatia(cbfdata, usedense=usedense)
        if hasintvars
            println("ignoring integrality constraints")
        end
        if !iszero(objoffset)
            println("ignoring objective offset")
        end

        println("constructing Hypatia model")
        Hypatia.check_data(c, A, b, G, h, cone)
        if lscachetype == "QRSymmCache"
            (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
            ls = Hypatia.QRSymmCache(c1, A1, b1, G1, h, cone, Q2, RiQ1)
        elseif lscachetype == "NaiveCache"
            (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=false)
            ls = Hypatia.NaiveCache(c1, A1, b1, G1, h, cone)
        end
        model = Hypatia.Model(; options...)
        Hypatia.load_data!(model, c1, A1, b1, G1, h, cone, ls)

        println("solving Hypatia model")
        try
            (val, runtime, bytes, gctime, memallocs) = @timed Hypatia.solve!(model)
            println("Hypatia finished")
            status = Hypatia.get_status(model)
            niters = model.niters
            pobj = Hypatia.get_pobj(model)
            dobj = Hypatia.get_dobj(model)
        catch solveerror
            println("Hypatia errored:")
            println(solveerror)
        end

        redirect_stdout(OUT)
        redirect_stderr(ERR)
    end

    if !isnothing(solveerror)
        println("Hypatia errored on $instname:")
        println(solveerror)
    end

    open(csvfile, "a") do fdcsv
        # TODO optionally save more information from memallocs
        println(fdcsv, "$instname,$status,$pobj,$dobj,$niters,$runtime,$gctime,$bytes")
    end
end
