using CSV
using DataFrames
using Plots
using Printf

MAX_TIME = 3600
MAX_ITER = 250

nickname = "various"

enhancements = ["basic", "TOA", "curve", "comb", "shift"]
process_entry(x::Float64) = @sprintf("%.2f", x)
process_entry(x::Int) = string(x)

bench_file = joinpath("bench2", "various", "bench_" * nickname * ".csv")

function shifted_geomean_all(metric, conv; shift = 0, cap = Inf)
    x = copy(metric)
    x[.!conv] .= cap
    return exp(sum(log, x .+ shift) / length(x)) - shift
end
shifted_geomean_conv(metric, conv; shift = 0) = exp(sum(log, metric[conv] .+ shift) / count(conv)) - shift

function post_process()
    all_df = CSV.read(bench_file, DataFrame)
    transform!(all_df,
        :status => ByRow(x -> !ismissing(x) && x in ["Optimal", "PrimalInfeasible", "DualInfeasible"]) => :conv,
        # each instance is identified by instance data + extender combination
        [:inst_data, :extender] => ((x, y) -> x .* y) => :inst_key,
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

    df_agg = combine(groupby(all_df, [:stepper, :system_solver, :use_corr, :use_curve_search, :shift]),
        [:solve_time, :conv] => shifted_geomean_conv => :time_geomean_thisconv,
        [:iters, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = 1)) => :iters_geomean_thisconv,
        [:solve_time, :all_conv] => shifted_geomean_conv => :time_geomean_allconv,
        [:iters, :all_conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = 1)) => :iters_geomean_allconv,
        [:solve_time, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME)) => :time_geomean_all,
        [:iters, :conv] => ((x, y) -> shifted_geomean_all(x, y, shift = 1, cap = MAX_ITER)) => :iters_geomean_all,
        :status => (x -> count(isequal("Optimal"), x)) => :optimal,
        :status => (x -> count(isequal("PrimalInfeasible"), x)) => :priminfeas,
        :status => (x -> count(isequal("DualInfeasible"), x)) => :dualinfeas,
        :status => (x -> count(isequal("NumericalFailure"), x)) => :numerical,
        :status => (x -> count(isequal("SlowProgress"), x)) => :slowprogress,
        :status => (x -> count(isequal("TimeLimit"), x)) => :timelimit,
        :status => (x -> count(isequal("IterationLimit"), x)) => :iterationlimit,
        :status => length => :total,
        )
    sort!(df_agg, [order(:stepper, rev = true), :use_corr, :use_curve_search, :shift, :system_solver])
    CSV.write(joinpath(output_folder, "agg_" * nickname * ".csv"), df_agg)

    subdfs = ["status", "iter", "time"]
    status = [:optimal, :priminfeas, :dualinfeas, :numerical, :slowprogress, :iterationlimit]
    iter = [:iters_geomean_thisconv, :iters_geomean_allconv, :iters_geomean_all]
    time = [:time_geomean_thisconv, :time_geomean_allconv, :time_geomean_all]
    sep = " & "

    for (k, subcols) in enumerate([status, iter, time])
        metric = subdfs[k]
        tex = open(joinpath(output_folder, metric * "_" * nickname * ".tex"), "w")
        subdf = df_agg[!, subcols]
        for i in 1:length(enhancements)
            row_str = enhancements[i]
            for j in 1:ncol(subdf)
                x = (metric == "time" ? subdf[i, j] * 1000 : subdf[i, j])
                row_str *= sep * process_entry(x)
            end
            row_str *= " \\\\"
            println(tex, row_str)
        end
        close(tex)

        # maybe merge table with above
        system_solver_tex = open(joinpath(output_folder, metric * "_system_solvers_" * nickname * ".tex"), "w")
        subdf = df_agg[!, subcols]
        for i in length(enhancements):nrow(subdf)
            row_str = ""
            for j in 1:ncol(subdf)
                x = (metric == "time" ? subdf[i, j] * 1000 : subdf[i, j])
                row_str *= sep * process_entry(x)
            end
            row_str *= " \\\\"
            println(system_solver_tex, row_str)
        end
        close(system_solver_tex)
    end

    return
end
agg_stats()

function bottlenecks()
    output_folder = mkpath(joinpath(@__DIR__, "results"))

    all_df = post_process()
    shift = 1e-4 # TODO use minimum values?

    preproc_cols = [:time_rescale, :time_initx, :time_inity, :time_unproc]

    df_agg = combine(groupby(all_df, [:stepper, :system_solver, :use_corr, :use_curve_search, :shift]),
        # vcat(:conv, preproc_cols) => ((y, x...) -> shifted_geomean_conv(sum(x), y, shift = shift)) => :preproc_thisconv,
        # [:time_upsys, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :uplhs_thisconv,
        # [:time_uprhs, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :uprhs_thisconv,
        # [:time_getdir, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :getdir_thisconv,
        # [:time_search, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :search_thisconv,
        # [:time_upsys, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :uplhs_piter_thisconv,
        # [:time_uprhs, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :uprhs_piter_thisconv,
        # [:time_getdir, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :getdir_piter_thisconv,
        # [:time_search, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :search_piter_thisconv,
        #
        vcat(:conv, preproc_cols) => ((y, x...) -> shifted_geomean_conv(sum(x), y, shift = shift)) => :preproc_allconv,
        [:time_upsys, :all_conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :uplhs_allconv,
        [:time_uprhs, :all_conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :uprhs_allconv,
        [:time_getdir, :all_conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :getdir_allconv,
        [:time_search, :all_conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :search_allconv,
        [:time_upsys, :all_conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :uplhs_piter_allconv,
        [:time_uprhs, :all_conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :uprhs_piter_allconv,
        [:time_getdir, :all_conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :getdir_piter_allconv,
        [:time_search, :all_conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :search_piter_allconv,
        # #
        # vcat(:conv, preproc_cols) => ((y, x...) -> shifted_geomean_all(sum(x), y, cap = MAX_TIME, shift = shift)) => :preproc_all,
        # [:time_upsys, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :uplhs_all,
        # [:time_uprhs, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :uprhs_all,
        # [:time_getdir, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :getdir_all,
        # [:time_search, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :search_all,
        # [:time_upsys, :conv, :iters] => ((x, y, z) -> shifted_geomean_all(x ./ (z .+ 1), y, cap = MAX_TIME, shift = shift)) => :uplhs_piter_all,
        # [:time_uprhs, :conv, :iters] => ((x, y, z) -> shifted_geomean_all(x ./ (z .+ 1), y, cap = MAX_TIME, shift = shift)) => :uprhs_piter_all,
        # [:time_getdir, :conv, :iters] => ((x, y, z) -> shifted_geomean_all(x ./ (z .+ 1), cap = MAX_TIME, y, shift = shift)) => :getdir_piter_all,
        # [:time_search, :conv, :iters] => ((x, y, z) -> shifted_geomean_all(x ./ (z .+ 1), cap = MAX_TIME, y, shift = shift)) => :search_piter_all,
        )
    sort!(df_agg, [order(:stepper, rev = true), :use_corr, :use_curve_search, :shift, :system_solver])
    CSV.write(joinpath(output_folder, "bottlenecks_" * nickname * ".csv"), df_agg)

    enhancements = ["basic", "TOA", "curve", "comb", "shift", "symdense", "symsparse"] # TODO make consistent with df_agg

    bottlenecks_tex = open(joinpath(output_folder, "bottlenecks_" * nickname * ".tex"), "w")
    sep = " & "
    for i in 1:nrow(df_agg)
        row_str = enhancements[i]
        for j in 6:ncol(df_agg)
            row_str *= sep * process_entry(df_agg[i, j] * 1000)
        end
        row_str *= " \\\\"
        println(bottlenecks_tex, row_str)
    end
    close(bottlenecks_tex)

    return
end
bottlenecks()

# performance profiles, currently hardcoded for corrector vs no corrector
function perf_prof(; feature = :stepper, metric = :solve_time)
    if feature == :stepper
        s1 = "PredOrCentStepper"
        s2 = "CombinedStepper"
        use_corr = [true]
        use_curve_search = [true]
        stepper = [s1, s2]
        shift = [0]
        system_solver = ["QRCholDenseSystemSolver"]
    elseif feature == :shift
        s1 = 0
        s2 = 2
        stepper = ["CombinedStepper"]
        use_corr = [true]
        use_curve_search = [true]
        system_solver = ["QRCholDenseSystemSolver"]
        shift = [s1, s2]
    elseif feature == :system_solver
        s1 = "QRCholDenseSystemSolver"
        s2 = "SymIndefSparseSystemSolver"
        stepper = ["CombinedStepper"]
        use_corr = [true]
        use_curve_search = [true]
        shift = [2]
        system_solver = [s1, s2]
    else
        # s1 = "FALSE"
        # s2 = "TRUE"
        s1 = false
        s2 = true
        stepper = ["PredOrCentStepper"]
        system_solver = ["QRCholDenseSystemSolver"]
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
        t.shift in shift &&
        t.system_solver in system_solver,
        all_df,
        )

    # remove instances where neither stepper being compared converged
    all_df = combine(groupby(all_df, :inst_key), names(all_df), :status => (x -> any(in(["Optimal", "PrimalInfeasible", "DualInfeasible"]).(x))) => :any_conv)
    filter!(t -> t.any_conv, all_df)

    select!(all_df,
        :inst_key,
        feature,
        [:solve_time, :conv] => ByRow((x, y) -> (y ? x : missing)) => :solve_time,
        [:iters, :conv] => ByRow((x, y) -> (y ? x : missing)) => :iters,
        )

    # replacing 0 iterations by 1 iteration gives correct result for all cases (none, one, or both steppers converged) and avoids NaNs
    transform!(all_df, :iters => ByRow(x -> !ismissing(x) && iszero(x) ? 1 : x) => :iters)

    all_df = combine(groupby(all_df, :inst_key), names(all_df), metric => (x -> x ./ minimum(skipmissing(x))) => :ratios)
    # assign maximum ratio to failures
    max_ratio = maximum(skipmissing(all_df[!, :ratios]))
    all_df[ismissing.(all_df[!, metric]), :ratios] .= max_ratio
    sort!(all_df, :ratios)

    npts = div(nrow(all_df), 2)
    subdf1 = filter(feature => isequal(s1), all_df)
    subdf2 = filter(feature => isequal(s2), all_df)
    unique_ratios = unique(all_df[!, :ratios])
    plot_x = log10.(unique_ratios)
    plot_y1 = [count(subdf1[!, :ratios] .<= ti) ./ npts for ti in unique_ratios]
    plot_y2 = [count(subdf2[!, :ratios] .<= ti) ./ npts for ti in unique_ratios]
    if metric == :solve_time
        @assert 1 <= plot_y1[1] + plot_y2[1] <= 1.01
    end

    plot(xlim = (0, log10(maximum(all_df[!, :ratios]))), ylim = (0, 1))
    plot!(log10.(unique_ratios), plot_y1, label = string(s1), t = :steppre)
    plot!(log10.(unique_ratios), plot_y2, label = string(s2), t = :steppre)
    xaxis!("logratio")
    plot_name = string(feature) * "_" * string(metric) * "_pp"
    title!(plot_name)

    png(plot_name)

    CSV.write(plot_name * ".csv", DataFrame(x = plot_x, y1 = plot_y1, y2 = plot_y2))

    return
end

for feature in [:stepper, :use_curve_search, :use_corr, :shift], metric in [:solve_time, :iters]
    @show feature, metric
    perf_prof(feature = feature, metric = metric)
end
# for feature in [:system_solver], metric in [:solve_time, :iters]
#     perf_prof(feature = feature, metric = metric)
# end

function modelstats()
    all_df = post_process()
    CSV.write("solvetimes.csv", DataFrame(hcat(log10.(all_df[!, :solve_time]))))

    one_solver = filter!(t ->
        t.stepper == "CombinedStepper" &&
        t.system_solver == "QRCholDenseSystemSolver" &&
        t.shift == 0,
        all_df
        )
    CSV.write("numcones.csv", DataFrame(hcat(one_solver[!, :num_cones])))
end
modelstats()

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

# boxplots
# using StatsPlots
# all_df = CSV.read(bench_file, DataFrame)
# transform!(all_df, [:inst_data, :extender] => ((x, y) -> x .* y) => :inst_key)
# all_df = combine(groupby(all_df, :inst_key), names(all_df), :status => (x -> all(isequal.("Optimal", x)) || all(isequal.("PrimalInfeasible", x))) => :all_conv)
# filter!(t -> t.all_conv == true, all_df)
# all_df = combine(groupby(all_df, :inst_key), names(all_df), :solve_time => sum => :time_sum)
# sort!(all_df, order(:time_sum, rev = true))
# all_df = all_df[1:(4 * 20), :]
# select!(all_df,
#     [:inst_data, :extender] => ((x, y) -> x .* y) => :k,
#     [:stepper, :use_corr, :use_curve_search] => ((a, b, c) -> a .* "_" .* string.(b) .* "_" .* string.(c)) => :stepper,
#     :solve_time => ByRow(log10) => :log_time,
#     # :solve_time,
#     )
# timings = unstack(all_df, :stepper, :log_time)
# # timings = unstack(all_df, :stepper, :solve_time)
# boxplot(["pc_00" "pc_01" "pc_11" "comb"], Matrix(timings[:, 2:end]), leg = false)

function stats_plots()
    all_df = post_process()
    histogram(log10.(all_df[!, :solve_time]))
    title!("log10 solve time")
    png("solvehist")

    histogram(log10.(sum(eachcol(all_df[!, [:n, :p, :q]]))))
    title!("log10 n + p + q")
    png("npqhist")

    histogram(log10.(max.(0.01, all_df[!, :n] - all_df[!, :p])))
    title!("log10(n - p)")
    png("nphist")

    histogram(all_df[!, :solve_time])
    title!("num cones")
    png("Khist")
end
# stats_plots()
