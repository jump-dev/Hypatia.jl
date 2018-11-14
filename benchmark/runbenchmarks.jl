#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

TODO readme file for benchmarks and describe ARGS for running on command line
=#

using Hypatia
# TODO replace with by ConicBenchmarkUtilities -> MOI -> Hypatia when CBU is updated for MOI
include(joinpath(@__DIR__, "Translate", "Translate.jl")) # module containing functions for translating from cbf to Hypatia native format

benchpath = @__DIR__
# benchpath = "/home/coey/.julia/dev/Hypatia/benchmark"

# parse command line arguments
# @assert length(ARGS) >= 2

cbfpath = "/home/coey/Dropbox/cblibeasy" # ARGS[1]
if !isdir(cbfpath)
    error("cbf path is not a valid directory: $cbfpath")
end

instanceset = "easy" # ARGS[2]
instsetfile = joinpath(benchpath, "instancesets", instanceset * ".txt")
if !isfile(instsetfile)
    error("instance set not found: instsetfile")
end
# check that each instance is in the cbfpath
instances = filter(l -> !isempty(l) && !startswith(l, '#'), strip.(readlines(instsetfile)))
@info "instance set $instanceset contains $(length(instances)) instances"
for instname in instances
    instfile = joinpath(cbfpath, instname * ".cbf.gz")
    if !isfile(instfile)
        error("could not find instance at $instfile")
    end
end

# timelimit = ARGS[3] # TODO add this

outputpath = "/home/coey/benchtest"
# @info "results will be saved in $outputfile"

lscachetype = "QRSymmCache" # ARGS[4]
if !in(lscachetype, ("QRSymmCache", "NaiveCache"))
    error("linear system cache type $lscachetype is not recognized")
end
usedense = parse(Bool, "false") # ARGS[5]


# Hypatia options
options = Dict()
# options[:maxiter] = 300
options[:verbose] = true


@info "starting benchmark run in a few seconds"
sleep(5.0)


# meta text file will contain information about the benchmarking process
fdmeta = open(joinpath(outputpath, "META_$(instanceset).txt"), "w")
println(fdmeta, "instance set $instanceset")
println(fdmeta, "$lscachetype for linear systems")
println(fdmeta, "$(usedense ? "dense" : "sparse") A and G")
println(fdmeta, "Hypatia options:")
for (k, v) in options
    println(fdmeta, "  $k = $v")
end

# each line of csv file will summarize Hypatia performance on a particular instance
fdcsv = open(joinpath(outputpath, "RESULTS_$(instanceset).csv"), "w")
println(fdcsv, "instancename,status,runtime,niters,pobj,dobj\n")

# run each instance, print Hypatia output to instance-specific file, and print metadata to a single csv file
TT = stdout
for instname in instances
    @info "starting instance $instname"

    fdinst = open(joinpath(outputpath, instname * ".txt"), "w")
    redirect_stdout(fdinst)

    println("reading CBF data and constructing Hypatia model")
    cbfdata = Translate.readcbfdata(joinpath(cbfpath, instname * ".cbf.gz"))
    (c, A, b, G, h, cone) = Translate.cbftohypatia(cbfdata, dense=usedense)
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

    noerror = true
    runtime = @elapsed try
        println("running Hypatia\n")
        Hypatia.solve!(model)
        println("\nHypatia finished")
        noerror = false
    catch e
        println("\nHypatia errored")
        println(e)
    end

    status = Hypatia.get_status(model)
    niters = model.niters
    pobj = Hypatia.get_pobj(model)
    dobj = Hypatia.get_dobj(model)

    redirect_stdout(TT)

    if !noerror
        @info "Hypatia errored on $instname"
    end

    println(fdcsv, "$instname,$status,$runtime,$niters,$pobj,$dobj\n")
end


close(fdmeta)
close(fdcsv)
