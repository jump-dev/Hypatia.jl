using CSV
using DataFrames
using Plots
using Printf
using BenchmarkProfiles

enhancements = ["basic", "TOA", "curve", "comb", "shift"]
process_entry(x::Float64) = @sprintf("%.2f", x)
process_entry(x::Int) = string(x)

bench_file = joinpath("bench2", "various", "bench" * ".csv")

function shifted_geomean(metric::AbstractVector, conv::AbstractVector{Bool}; shift = 0, cap = -1, skipnotfinite = false)
    if cap > 0
        x = copy(metric)
        x[.!conv] .= cap
    else
        x = metric[conv]
    end
    if skipnotfinite
        x = x[isfinite.(x)]
    end
    return exp(sum(log, x .+ shift) / length(x)) - shift
end

function post_process()
    all_df = CSV.read(bench_file, DataFrame)
    transform!(all_df,
        :status => ByRow(x -> !ismissing(x) && x in ["Optimal", "PrimalInfeasible", "DualInfeasible"]) => :conv,
        # each instance is identified by instance data + extender combination
        [:example, :inst_data, :extender] => ((x, y, z) -> x .* y .* z) => :inst_key,
        )
    # assumes that nothing returned incorrect status, which is checked manually
    all_df = combine(groupby(all_df, :inst_key), names(all_df), :status => (x -> all(in(["Optimal", "PrimalInfeasible", "DualInfeasible"]).(x))) => :all_conv)
    # remove precompile instances
    filter!(t -> t.inst_set == "various", all_df)
    return all_df
end


# aggregate stuff
function agg_stats()
    output_folder = mkpath(joinpath(@__DIR__, "results"))

    all_df = post_process()
    MAX_TIME = maximum(all_df[!, :solve_time])
    MAX_ITER = maximum(all_df[!, :iters])

    df_agg = combine(groupby(all_df, [:stepper, :use_corr, :use_curve_search, :shift]),
        [:solve_time, :conv] => shifted_geomean => :time_geomean_thisconv,
        [:iters, :conv] => ((x, y) -> shifted_geomean(x, y, shift = 1)) => :iters_geomean_thisconv,
        [:solve_time, :all_conv] => shifted_geomean => :time_geomean_allconv,
        [:iters, :all_conv] => ((x, y) -> shifted_geomean(x, y, shift = 1)) => :iters_geomean_allconv,
        [:solve_time, :conv] => ((x, y) -> shifted_geomean(x, y, cap = MAX_TIME)) => :time_geomean_all,
        [:iters, :conv] => ((x, y) -> shifted_geomean(x, y, shift = 1, cap = MAX_ITER)) => :iters_geomean_all,
        :status => (x -> count(isequal("Optimal"), x)) => :optimal,
        :status => (x -> count(isequal("PrimalInfeasible"), x)) => :priminfeas,
        :status => (x -> count(isequal("DualInfeasible"), x)) => :dualinfeas,
        :status => (x -> count(isequal("NumericalFailure"), x)) => :numerical,
        :status => (x -> count(isequal("SlowProgress"), x)) => :slowprogress,
        :status => (x -> count(isequal("TimeLimit"), x)) => :timelimit,
        :status => (x -> count(isequal("IterationLimit"), x)) => :iterationlimit,
        :status => length => :total,
        )
    sort!(df_agg, [order(:stepper, rev = true), :use_corr, :use_curve_search, :shift])
    CSV.write(joinpath(output_folder, "agg" * ".csv"), df_agg)

    # combine feasible and infeasible statuses
    transform!(df_agg, [:optimal, :priminfeas, :dualinfeas] => ByRow((x...) -> sum(x)) => :converged)
    cols = [:converged, :iters_geomean_thisconv, :iters_geomean_allconv, :iters_geomean_all, :time_geomean_thisconv, :time_geomean_allconv, :time_geomean_all]
    sep = " & "
    tex = open(joinpath(output_folder, "agg" * ".tex"), "w")
    for i in 1:length(enhancements)
        row_str = enhancements[i]
        for c in cols
            subdf = df_agg[!, c]
            row_str *= sep * process_entry(subdf[i])
        end
        row_str *= " \\\\"
        println(tex, row_str)
    end
    close(tex)

    return
end
# agg_stats()

function subtime()
    output_folder = mkpath(joinpath(@__DIR__, "results"))

    all_df = post_process()
    shift = 1e-4

    preproc_cols = [:time_rescale, :time_initx, :time_inity, :time_unproc]
    metrics = [:preproc, :uplhs, :uprhs, :getdir, :search, :uplhs_piter, :uprhs_piter, :getdir_piter, :search_piter]
    sets = [:_thisconv, :_allconv, :_all]

    # get values to replace unconverged instances for the "all" group
    max_upsys = maximum(all_df[!, :time_upsys])
    max_uprhs =  maximum(all_df[!, :time_uprhs])
    max_getdir =  maximum(all_df[!, :time_getdir])
    max_search =  maximum(all_df[!, :time_search])
    max_upsys_iter =  maximum(x -> isfinite(x) ? x : 0, all_df[!, :time_upsys] ./ all_df[!, :iters])
    max_uprhs_iter =  maximum(x -> isfinite(x) ? x : 0, all_df[!, :time_uprhs] ./ all_df[!, :iters])
    max_getdir_iter =  maximum(x -> isfinite(x) ? x : 0, all_df[!, :time_getdir] ./ all_df[!, :iters])
    max_search_iter =  maximum(x -> isfinite(x) ? x : 0, all_df[!, :time_search] ./ all_df[!, :iters])

    df_agg = combine(groupby(all_df, [:stepper, :use_corr, :use_curve_search, :shift]),
        vcat(:conv, preproc_cols) => ((y, x...) -> shifted_geomean(sum(x), y, shift = shift)) => :preproc_thisconv,
        [:time_upsys, :conv] => ((x, y) -> shifted_geomean(x, y, shift = shift)) => :uplhs_thisconv,
        [:time_uprhs, :conv] => ((x, y) -> shifted_geomean(x, y, shift = shift)) => :uprhs_thisconv,
        [:time_getdir, :conv] => ((x, y) -> shifted_geomean(x, y, shift = shift)) => :getdir_thisconv,
        [:time_search, :conv] => ((x, y) -> shifted_geomean(x, y, shift = shift)) => :search_thisconv,
        [:time_upsys, :conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, shift = shift, skipnotfinite = true)) => :uplhs_piter_thisconv,
        [:time_uprhs, :conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, shift = shift, skipnotfinite = true)) => :uprhs_piter_thisconv,
        [:time_getdir, :conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, shift = shift, skipnotfinite = true)) => :getdir_piter_thisconv,
        [:time_search, :conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, shift = shift, skipnotfinite = true)) => :search_piter_thisconv,
        #
        vcat(:all_conv, preproc_cols) => ((y, x...) -> shifted_geomean(sum(x), y, shift = shift)) => :preproc_allconv,
        [:time_upsys, :all_conv] => ((x, y) -> shifted_geomean(x, y, shift = shift)) => :uplhs_allconv,
        [:time_uprhs, :all_conv] => ((x, y) -> shifted_geomean(x, y, shift = shift)) => :uprhs_allconv,
        [:time_getdir, :all_conv] => ((x, y) -> shifted_geomean(x, y, shift = shift)) => :getdir_allconv,
        [:time_search, :all_conv] => ((x, y) -> shifted_geomean(x, y, shift = shift)) => :search_allconv,
        [:time_upsys, :all_conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, shift = shift, skipnotfinite = true)) => :uplhs_piter_allconv,
        [:time_uprhs, :all_conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, shift = shift, skipnotfinite = true)) => :uprhs_piter_allconv,
        [:time_getdir, :all_conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, shift = shift, skipnotfinite = true)) => :getdir_piter_allconv,
        [:time_search, :all_conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, shift = shift, skipnotfinite = true)) => :search_piter_allconv,
        #
        vcat(:conv, preproc_cols) => ((y, x...) -> shifted_geomean(sum(x), y, cap = max_upsys, shift = shift)) => :preproc_all,
        [:time_upsys, :conv] => ((x, y) -> shifted_geomean(x, y, cap = max_upsys, shift = shift)) => :uplhs_all,
        [:time_uprhs, :conv] => ((x, y) -> shifted_geomean(x, y, cap = max_uprhs, shift = shift)) => :uprhs_all,
        [:time_getdir, :conv] => ((x, y) -> shifted_geomean(x, y, cap = max_getdir, shift = shift)) => :getdir_all,
        [:time_search, :conv] => ((x, y) -> shifted_geomean(x, y, cap = max_search, shift = shift)) => :search_all,
        [:time_upsys, :conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, cap = max_upsys_iter, shift = shift, skipnotfinite = true)) => :uplhs_piter_all,
        [:time_uprhs, :conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, y, cap = max_uprhs_iter, shift = shift, skipnotfinite = true)) => :uprhs_piter_all,
        [:time_getdir, :conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, cap = max_getdir_iter, y, shift = shift, skipnotfinite = true)) => :getdir_piter_all,
        [:time_search, :conv, :iters] => ((x, y, z) -> shifted_geomean(x ./ z, cap = max_search_iter, y, shift = shift, skipnotfinite = true)) => :search_piter_all,
        )
    sort!(df_agg, [order(:stepper, rev = true), :use_corr, :use_curve_search, :shift])
    CSV.write(joinpath(output_folder, "subtime" * ".csv"), df_agg)

    subtime_tex = open(joinpath(output_folder, "subtime" * ".tex"), "w")
    sep = " & "

    for s in sets, i in 1:nrow(df_agg)
        row_str = sep * enhancements[i]
        for m in metrics
            col = Symbol(m, s)
            row_str *= sep * process_entry(df_agg[i, col] * 1000)
        end
        row_str *= " \\\\"
        println(subtime_tex, row_str)
    end

    close(subtime_tex)

    return
end
subtime()

# performance profiles, currently hardcoded for corrector vs no corrector
function perf_prof(; feature = :stepper, metric = :solve_time)
    if feature == :stepper
        s1 = "PredOrCentStepper"
        s2 = "CombinedStepper"
        use_corr = [true]
        use_curve_search = [true]
        stepper = [s1, s2]
        shift = [0]
    elseif feature == :shift
        s1 = 0
        s2 = 2
        stepper = ["CombinedStepper"]
        use_corr = [true]
        use_curve_search = [true]
        shift = [s1, s2]
    else
        s1 = false
        s2 = true
        stepper = ["PredOrCentStepper"]
        shift = [0]
        if feature == :use_corr
            use_curve_search = [false]
            use_corr = [s1, s2]
        elseif feature == :use_curve_search
            use_corr = [true]
            use_curve_search = [s1, s2]
        end
    end

    all_df = post_process()
    filter!(t ->
        t.stepper in stepper &&
        t.use_corr in use_corr &&
        t.use_curve_search in use_curve_search &&
        t.shift in shift,
        all_df
        )

    # remove instances where neither stepper being compared converged
    # all_df = combine(groupby(all_df, :inst_key), names(all_df), :conv => any => :any_conv)
    # filter!(t -> t.any_conv, all_df)

    # BenchmarkProfiles expects NaNs for failures
    select!(all_df,
        :inst_key,
        feature,
        [metric, :conv] => ByRow((x, y) -> (y ? x : NaN)) => metric,
        )

    wide_df = unstack(all_df, feature, metric)
    @show names(wide_df)
    (ratios, max_ratio) = BenchmarkProfiles.performance_ratios(Matrix{Float64}(wide_df[!, string.([s1, s2])]))

    # copied from https://github.com/JuliaSmoothOptimizers/BenchmarkProfiles.jl/blob/master/src/performance_profiles.jl
    (np, ns) = size(ratios)
    ratios = [ratios; 2.0 * max_ratio * ones(1, ns)]
    xs = [1:np+1;] / np
    for s = 1 : 2
        rs = view(ratios, :, s)
        xidx = zeros(Int,length(rs)+1)
        k = 0
        rv = minimum(rs)
        maxval = maximum(rs)
        while rv < maxval
            k += 1
            xidx[k] = findlast(rs .<= rv)
            rv = max(rs[xidx[k]], rs[xidx[k]+1])
        end
        xidx[k+1] = length(rs)
        xidx = xidx[xidx .> 0]
        xidx = unique(xidx) # Needed?

        # because we are making a step line graph from a CSV, modify (x, y) to make steps
        x = rs[xidx]
        y = xs[xidx]
        x = vcat(x[1], repeat(x[2:end], inner = 2))
        y = vcat(repeat(y[1:(end - 1)], inner = 2), y[end])
        CSV.write(string(feature) * "_" * string(metric) * "_$(s)" * "_pp" * ".csv", DataFrame(x = x, y = y))
    end

    return
end

for feature in [:stepper, :use_curve_search, :use_corr, :shift], metric in [:solve_time, :iters]
    @show feature, metric
    perf_prof(feature = feature, metric = metric)
end

function instancestats()
    all_df = post_process()

    one_solver = filter!(t ->
        t.stepper == "PredOrCentStepper" &&
        t.use_corr == 0 &&
        t.use_curve_search == 0,
        all_df
        )
    inst_df = select(one_solver, :num_cones => ByRow(log10) => :numcones, [:n, :p, :q] => ((x, y, z) -> log10.(x .+ y .+ z)) => :npq)
    CSV.write("inststats.csv", inst_df)
    # for solve times, only include converged instances
    solve_times = filter!(t -> t.conv, one_solver)
    CSV.write("solvetimes.csv", select(solve_times, :solve_time => ByRow(log10) => :time))

    # only used to get list of cones manually
    ex_df = combine(groupby(one_solver, :example),
        :cone_types => (x -> union(eval.(Meta.parse.(x)))) => :cones,
        :cone_types => length => :num_instances,
        )
    CSV.write("examplestats.csv", ex_df)

    return
end
instancestats()

# comb_df = filter(:stepper => isequal("CombinedStepper"), dropmissing(all_df))
# pc_df = filter(t -> t.stepper == "PredOrCentStepper" && t.use_curve_search == true, dropmissing(all_df))
# comb_times = (comb_df[!, :solve_time])
# pc_times = (pc_df[!, :solve_time])
#
# plot(comb_times, seriestype=:stephist)
# plot!(pc_times, seriestype=:stephist)

# missig instances
# all_df = CSV.read(bench_file, DataFrame)
# insts1 = filter(t -> t.stepper == "PredOrCentStepper" && t.use_curve_search == true, dropmissing(all_df))
# insts2 = filter(t -> t.stepper == "CombinedStepper" && t.shift == 1, dropmissing(all_df))
# @show setdiff(unique(insts2[!, :inst_data]), unique(insts1[!, :inst_data]))

# # boxplots
# using StatsPlots
# all_df = post_process()
# transform!(all_df, [:inst_data, :extender] => ((x, y) -> x .* y) => :inst_key)
# all_df = combine(groupby(all_df, :inst_key), names(all_df), :status => (x -> all(isequal.("Optimal", x)) || all(isequal.("PrimalInfeasible", x))) => :all_conv)
# filter!(t -> t.all_conv, all_df)
# filter!(t -> t.shift == 0, all_df)
# select!(all_df,
#     [:inst_data, :extender] => ((x, y) -> x .* y) => :k,
#     [:stepper, :use_corr, :use_curve_search] => ((a, b, c) -> a .* "_" .* string.(b) .* "_" .* string.(c)) => :stepper,
#     :solve_time => ByRow(log10) => :log_time,
#     )
# timings = unstack(all_df, :stepper, :log_time)
# boxplot(["pc_00" "pc_01" "pc_11" "comb"], Matrix(timings[:, 2:end]), leg = false)

# function stats_plots()
#     all_df = post_process()
#     histogram(log10.(all_df[!, :solve_time]))
#     title!("log10 solve time")
#     png("solvehist")
#
#     histogram(log10.(sum(eachcol(all_df[!, [:n, :p, :q]]))))
#     title!("log10 n + p + q")
#     png("npqhist")
#
#     histogram(log10.(max.(0.01, all_df[!, :n] - all_df[!, :p])))
#     title!("log10(n - p)")
#     png("nphist")
#
#     histogram(all_df[!, :solve_time])
#     title!("num cones")
#     png("Khist")
# end
# stats_plots()
