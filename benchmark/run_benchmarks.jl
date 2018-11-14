#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using Hypatia
using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles

benchmark_dir = joinpath(dirname(dirname(pathof(Hypatia))), "benchmark")

# Module for translation, can be replaced by CBU when updated
include(joinpath(benchmark_dir, "Translate", "Translate.jl"))

# @assert length(ARGS[1]) >= 2
path_to_input = "D:/cblib/cblib.zib.de/download/all"  # ARGS[1]
set_to_run = "easy" # ARGS[2]
# time_limit = ARGS[3] # TODO add this

if !isdir(path_to_input)
    error("Not a valid input directory.")
end
setfile = joinpath(benchmark_dir, "instancesets", set_to_run*".txt")
if !isfile(setfile)
    error("Not a valid input directory.")
end

# Output
write_output = true
results_file_name = joinpath(@__DIR__(), "benchmark_times_$(round(Int, time())).csv")

hypatia_cachetypes = ["qrsym"]

# Hypatia options
mdl_options = Dict()
mdl_options[:maxiter] = 300
mdl_options[:verbose] = true

# For constructing CBLIB problems
function get_cblib_data(problem::String)
    @info "Reading data"
    return Translate.readcbfdata(joinpath(path_to_input, problem))
end

function cblib_data_mdl(dat::Translate.CBFData, lscachetype::String, kwargs::Dict)
    (c, A, b, G, h, cone) = Translate.cbftohypatia(dat, dense=false)
    Hypatia.check_data(c, A, b, G, h, cone)
    if lscachetype == "qrsym"
        (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
        L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, cone, Q2, RiQ1)
    elseif lscachetype == "naive"
        (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=false)
        L = Hypatia.NaiveCache(c1, A1, b1, G1, h, cone)
    else
        error("linear system cache type $lscachetype is not recognized")
    end
    mdl = Hypatia.Model(; kwargs...)
    Hypatia.load_data!(mdl, c1, A1, b1, G1, h, cone, L)
    return (c, A, b, G, h, cone, prkeep, dukeep, mdl)
end

function check_certificates(c, A, b, G, h, cone, prkeep, dukeep, mdl)
    # TODO 
    return true
end

function benchmarksolve!(mdl::Hypatia.Model, problem::String)
    try
        Hypatia.solve!(mdl)
        status = Hypatia.get_status(mdl)
        if status != :Optimal
            open(joinpath(@__DIR__(), "error_log.txt"), "a") do f
                println(f, problem, ",", string(status))
            end
        end
    catch e
        open(joinpath(@__DIR__(), "error_log.txt"), "a") do f
            println(f, problem, ",", e)
        end
    end
end

function result(problem::String, cache::String, id::Int, tm::Float64, cert::Bool, mdl::Hypatia.Model)
    niters = mdl.niters
    status = Hypatia.get_status(mdl)
    if status != :StartedIterating
        pobj = Hypatia.get_pobj(mdl)
        dobj = Hypatia.get_dobj(mdl)
    else
        pobj = Inf
        dobj = -Inf
    end
    (problem=problem, cahce=cache, id=id, time=tm, cert=cert, niters=niters, status=status, pobj=pobj, dobj=dobj)
end

function run_instances()

    problems = readdlm(setfile, comments=true)
    @assert size(problems, 2) == 1

    if write_output
        open(results_file_name, "w") do f
            write(f, "#")
            for (k, v) in mdl_options
                write(f, " $k = $v")
            end
            write(f, "\nproblem,id,cache,hypatia_time,hypatia_iterations,pobj,dobj,certificate,status\n")
        end
    end

    results = []

    for i in 1:size(problems, 1)
        problem_name = problems[i, 1] * ".cbf.gz"
        problem_file = joinpath(path_to_input, problem_name)
        @info "Testing problem: $problem_name"

        for cache in hypatia_cachetypes
            @info "Testing with $cache"
            dat = get_cblib_data(problem_file)
            (c, A, b, G, h, cone, prkeep, dukeep, mdl) = cblib_data_mdl(dat, cache, mdl_options)
            @info "Solving in Hypatia"
            htime = @elapsed benchmarksolve!(mdl, problem_name)
            cert = check_certificates(c, A, b, G, h, cone, prkeep, dukeep, mdl)
            @info "Checking certificates"
            r = result(problem_name, cache, i, htime, cert, mdl)
            push!(results, r)

            if write_output
                open(results_file_name, "a") do f
                    write(f, "$problem_name,$i,$cache, $htime, $(r.niters), $(r.pobj), $(r.dobj),$cert,$(r.status)\n")
                end
            end
        end # cache types
    end # problems

    return results
end

run_instances()
