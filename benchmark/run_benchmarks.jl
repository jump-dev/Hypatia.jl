#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using Hypatia
using Random
using LinearAlgebra
using SparseArrays
using DelimitedFiles

Random.seed!(32)

# Module for translation, can be replaced by CBU
benchmark_dir = joinpath(dirname(dirname(pathof(Hypatia))), "benchmark")
include(joinpath(benchmark_dir, "Translate", "Translate.jl"))

# Either use a directory of downloaded instances or download them on the fly
dflt_cbf_dir = "D:/cblib/cblib.zib.de/download/all"

# Output
write_output = true
results_file_name = joinpath(@__DIR__(), "benchmark_times_$(round(Int, time())).csv")

# Problems of interest
max_size = 100_000_000.0
skip_cones = []
hypatia_supported_cones = ["orthant", "exp", "soc", "rsoc", "psd", "sos","power"]
hypatia_cachetypes = ["qrsym"]
skip_problems = [
    "as_conic_100_100_hard_set1_1_cap10.cbf.gz" # stalls on these problems
    "chainsing-1000-3.cbf.gz" # took 3 hours then predictor fail
    "uflquad-nopsc-20-100.cbf.gz" # taking a long time
]
# If concerned with certain instances
only_chosen_problems = true
chosen_problems = ["2x5_1scen_8bars.cbf.gz"]
start_index = 0
# Versions to test
mdl_options = Dict()
mdl_options[:maxiter] = 300
mdl_options[:verbose] = true

# For constructing CBLIB problems
function get_cblib_data(problem::String)
    @info "Reading data"
    if isdir(dflt_cbf_dir)
        return Translate.readcbfdata(joinpath(dflt_cbf_dir, problem))
    else
        run(`wget -l1 -np http://cblib.zib.de/download/all/$problem`)
        dat = Translate.readcbfdata(problem)
        rm(problem)
        return dat
    end
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

function relevant(datarow, header)
    for c in skip_cones
        if datarow[header[:] .== c][1]
            return false
        end
    end
    if only_chosen_problems
        if !(datarow[1] in chosen_problems)
            return false
        end
    end
    if datarow[1] in skip_problems
        return false
    end
    # Some families we want to avoid
    if startswith(datarow[1], "as_conic_100_100_hard")
        return false
    end
    if startswith(datarow[1], "chainsing")
        return false
    end
    if datarow[2] < start_index
        return false
    end
    return datarow[10] < max_size
end

# saved details about each problem
problem_stats = readdlm(joinpath(@__DIR__(), "data/cblib_problem_stats.csv"), ',', header = true)

function check_certificates(c, A, b, G, h, cone, prkeep, dukeep, mdl)
    status = Hypatia.get_status(mdl)
    # If we fail due to a numerical issue skip this, error should be logged
    if status == :StartedIterating
        return false
    end
    (atol, rtol) = (1e-4, 1e-4)
    # construct solution
    x = zeros(length(c))
    x[dukeep] = Hypatia.get_x(mdl)
    y = zeros(length(b))
    y[prkeep] = Hypatia.get_y(mdl)
    s = Hypatia.get_s(mdl)
    z = Hypatia.get_z(mdl)
    pobj = Hypatia.get_pobj(mdl)
    dobj = Hypatia.get_dobj(mdl)

    ret = true

    # check conic certificates are valid; conditions are described by CVXOPT at https://github.com/cvxopt/cvxopt/blob/master/src/python/coneprog.py
    if status == :Optimal
        isapprox(pobj, dobj, atol=atol, rtol=rtol) || (ret = false)
        isapprox(A*x, b, atol=atol, rtol=rtol) || (ret = false)
        isapprox(G*x + s, h, atol=atol, rtol=rtol) || (ret = false)
        isapprox(G'*z + A'*y, -c, atol=atol, rtol=rtol) || (ret = false)
        isapprox(dot(s, z), 0.0, atol=atol, rtol=rtol) || (ret = false)
        isapprox(dot(c, x), pobj, atol=1e-8, rtol=1e-8) || (ret = false)
        isapprox(dot(b, y) + dot(h, z), -dobj, atol=1e-8, rtol=1e-8) || (ret = false)
    elseif status == :PrimalInfeasible
        (isnan(pobj)) || (ret = false)
        isapprox(dobj, 1.0, atol=1e-8, rtol=1e-8) || (ret = false)
        (all(isnan, x)) || (ret = false)
        (all(isnan, s)) || (ret = false)
        isapprox(dot(b, y) + dot(h, z), -1.0, atol=1e-8, rtol=1e-8) || (ret = false)
        isapprox(G'*z, -A'*y, atol=atol, rtol=rtol) || (ret = false)
    elseif status == :DualInfeasible
        (isnan(dobj)) || (ret = false)
        isapprox(pobj, -1.0, atol=1e-8, rtol=1e-8) || (ret = false)
        (all(isnan, y)) || (ret = false)
        (all(isnan, z)) || (ret = false)
        isapprox(dot(c, x), -1.0, atol=1e-8, rtol=1e-8) || (ret = false)
        isapprox(G*x, -s, atol=atol, rtol=rtol) || (ret = false)
        isapprox(A*x, zeros(length(y)), atol=atol, rtol=rtol) || (ret = false)
    elseif status == :IllPosed
        (all(isnan, x)) || (ret = false)
        (all(isnan, s)) || (ret = false)
        (all(isnan, y)) || (ret = false)
        (all(isnan, z)) || (ret = false)
    end
    return ret
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

function result(problem::String, cache::String, id::Int, size::Float64, tm::Float64, cert::Bool, mdl::Hypatia.Model)
    niters = mdl.niters
    status = Hypatia.get_status(mdl)
    if status != :StartedIterating
        pobj = Hypatia.get_pobj(mdl)
        dobj = Hypatia.get_dobj(mdl)
    else
        pobj = Inf
        dobj = -Inf
    end
    (problem=problem, cahce=cache, id=id, size=size, time=tm, cert=cert, niters=niters, status=status, pobj=pobj, dobj=dobj)
end

function run_instances()

    if write_output
        open(results_file_name, "w") do f
            write(f, "#")
            for (k, v) in mdl_options
                write(f, " $k = $v")
            end
            write(f, "\nproblem,id,cache,size,hypatia_time,hypatia_iterations,pobj,dobj,certificate,status\n")
        end
    end

    results = []

    for i in 1:size(problem_stats[1], 1)
        row = problem_stats[1][i, :]
        problem = string(row[1])
        @assert endswith(problem, ".cbf.gz")
        # file size in kilobytes
        kb = problem_stats[1][i, 10] / 1000.0
        cones_involved = problem_stats[2][row .== true]
        skip = false
        # Cones in the problem. Last boolean is a flag for integers.
        for c in cones_involved[1:end-1]
            c in hypatia_supported_cones || (skip = true)
        end
        # Only use problems that were sampled from large families
        sampled_problems[i] || (skip = true)

        for cache in hypatia_cachetypes
            if !relevant(row, problem_stats[2]) || skip
                @info "Skipping problem $i: $problem"
                htime = 9999.0
                niters = 9999.0
                cert = false
                status = :Skipped
                r = (problem=problem, cache="NoCache", id=i, size=kb, time=htime, cert=cert, niters=niters, status=status, pobj=Inf, dobj=-Inf)
            else
                @info "Testing problem $i: $problem"
                @info "Testing with $cache"
                dat = get_cblib_data(problem)
                (c, A, b, G, h, cone, prkeep, dukeep, mdl) = cblib_data_mdl(dat, cache, mdl_options)
                @info "Solving in Hypatia"
                htime = @elapsed benchmarksolve!(mdl, problem)
                cert = check_certificates(c, A, b, G, h, cone, prkeep, dukeep, mdl)
                @info "Checking certificates"
                r = result(problem, cache, i, float(kb), htime, cert, mdl)
            end
            push!(results, r)

            if write_output
                open(results_file_name, "a") do f
                    write(f, "$problem,$i,$cache,$kb, $htime, $(r.niters), $(r.pobj), $(r.dobj),$cert,$(r.status)\n")
                end
            end
        end # cache types
    end # problems

    return results
end

run_instances()
