using CSV
using DataFrames
using Plots

MAX_TIME = 3600
MAX_ITER = 250

nickname = "various"

bench_file = joinpath("bench2", "various", "bench_" * nickname * ".csv")

function shifted_geomean_all(metric, conv; shift = 0, cap = Inf)
    x = copy(metric)
    x[.!conv] .= cap
    return exp(sum(log, x .+ shift) / length(x))
end
shifted_geomean_conv(metric, conv; shift = 0) = exp(sum(log, metric[conv] .+ shift) / count(conv))
shifted_geomean_all_conv(metric, all_conv; shift = 0) = exp(sum(log, metric[all_conv] .+ shift) / count(all_conv))

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

    df_agg = combine(groupby(all_df, [:stepper, :use_corr, :use_curve_search, :shift]),
        [:solve_time, :conv] => shifted_geomean_conv => :time_geomean_thisconv,
        [:iters, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = 1)) => :iters_geomean_thisconv,
        [:solve_time, :all_conv] => shifted_geomean_all_conv => :time_geomean_allconv,
        [:iters, :all_conv] => ((x, y) -> shifted_geomean_all_conv(x, y, shift = 1)) => :iters_geomean_allconv,
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
    sort!(df_agg, [order(:stepper, rev = true), :use_corr, :use_curve_search, :shift])
    CSV.write(joinpath(output_folder, "df_agg_" * nickname * ".csv"), df_agg)

    return
end
agg_stats()

function bottlenecks()
    output_folder = mkpath(joinpath(@__DIR__, "results"))

    all_df = post_process()
    shift = 1e-4

    df_agg = combine(groupby(all_df, [:stepper, :use_corr, :use_curve_search, :shift]),
        [:time_rescale, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :rescale_thisconv,
        [:time_initx, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :initx_thisconv,
        [:time_inity, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :inity_thisconv,
        [:time_unproc, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :unproc_thisconv,
        [:time_uplhs, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :uplhs_thisconv,
        [:time_uprhs, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :uprhs_thisconv,
        [:time_getdir, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :getdir_thisconv,
        [:time_search, :conv] => ((x, y) -> shifted_geomean_conv(x, y, shift = shift)) => :search_thisconv,
        #
        [:time_rescale, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :rescale_piter_thisconv,
        [:time_initx, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :initx_piter_thisconv,
        [:time_inity, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :inity_piter_thisconv,
        [:time_unproc, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :unproc_piter_thisconv,
        [:time_uplhs, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :uplhs_piter_thisconv,
        [:time_uprhs, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :uprhs_piter_thisconv,
        [:time_getdir, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :getdir_piter_thisconv,
        [:time_search, :conv, :iters] => ((x, y, z) -> shifted_geomean_conv(x ./ (z .+ 1), y, shift = shift)) => :search_piter_thisconv,
        #
        [:time_rescale, :all_conv] => ((x, y) -> shifted_geomean_all_conv(x, y, shift = shift)) => :rescale_allconv,
        [:time_initx, :conv] => ((x, y) -> shifted_geomean_all_conv(x, y, shift = shift)) => :initx_allconv,
        [:time_inity, :conv] => ((x, y) -> shifted_geomean_all_conv(x, y, shift = shift)) => :inity_allconv,
        [:time_unproc, :conv] => ((x, y) -> shifted_geomean_all_conv(x, y, shift = shift)) => :unproc_allconv,
        [:time_uplhs, :conv] => ((x, y) -> shifted_geomean_all_conv(x, y, shift = shift)) => :uplhs_allconv,
        [:time_uprhs, :conv] => ((x, y) -> shifted_geomean_all_conv(x, y, shift = shift)) => :uprhs_allconv,
        [:time_getdir, :conv] => ((x, y) -> shifted_geomean_all_conv(x, y, shift = shift)) => :getdir_allconv,
        [:time_search, :conv] => ((x, y) -> shifted_geomean_all_conv(x, y, shift = shift)) => :search_allconv,
        #
        [:time_rescale, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :rescale_all,
        [:time_initx, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :initx_all,
        [:time_inity, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :inity_all,
        [:time_unproc, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :unproc_all,
        [:time_uplhs, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :uplhs_all,
        [:time_uprhs, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :uprhs_all,
        [:time_getdir, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :getdir_all,
        [:time_search, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME, shift = shift)) => :search_all,
        )
    sort!(df_agg, [order(:stepper, rev = true), :use_corr, :use_curve_search, :shift])
    CSV.write(joinpath(output_folder, "df_bottlenecks_" * nickname * ".csv"), df_agg)

    return
end
bottlenecks()

# performance profiles, currently hardcoded for corrector vs no corrector
function perf_prof(; feature = :stepper, metric = :solve_time)
    if feature == :stepper
        s1 = "Hypatia.Solvers.PredOrCentStepper{Float64}"
        s2 = "Hypatia.Solvers.CombinedStepper{Float64}"
        use_corr = [true]
        use_curve_search = [true]
        stepper = [s1, s2]
        shift = [0]
    elseif feature == :shift
        s1 = 0
        s2 = 2
        use_corr = [true]
        use_curve_search = [true]
        stepper = ["Hypatia.Solvers.CombinedStepper{Float64}"]
        shift = [s1, s2]
    else
        # s1 = "FALSE"
        # s2 = "TRUE"
        s1 = false
        s2 = true
        stepper = ["Hypatia.Solvers.PredOrCentStepper{Float64}"]
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
        all_df,
        )

    # remove instances where neither stepper being compared converged
    all_df = combine(groupby(all_df, :inst_key), names(all_df), :status => (x -> any(in(["Optimal", "PrimalInfeasible", "DualInfeasible"]).(x))) => :any_conv)
    filter!(t -> t.any_conv, all_df)

    select!(all_df,
        :inst_key,
        feature,
        [:solve_time, :conv] => ByRow((x, y) -> (y ? x : NaN)) => :solve_time,
        [:iters, :conv] => ByRow((x, y) -> (y ? x : NaN)) => :iters,
        )

    all_df = combine(groupby(all_df, :inst_key), names(all_df), metric => (x -> x ./ minimum(x)) => :ratios)
    # assign maximum ratio to failures
    max_ratio = maximum(filter(!isnan, all_df[!, :ratios]))
    all_df[isnan.(all_df[!, metric]), :ratios] .= max_ratio
    sort!(all_df, :ratios)

    nsolvers = 2
    npts = nrow(all_df)

    plot(xlim = (0, log10(maximum(all_df[!, :ratios])) + 1), ylim = (0, 1))
    subdf = all_df[all_df[!, feature] .== s1, :]
    plot!(log10.(all_df[!, :ratios]), [sum(subdf[!, :ratios] .<= ti) ./ npts * nsolvers for ti in all_df[!, :ratios]], label = string(s1), t = :steppre)

    subdf = all_df[all_df[!, feature] .== s2, :]
    plot!(log10.(all_df[!, :ratios]), [sum(subdf[!, :ratios] .<= ti) ./ npts * nsolvers for ti in all_df[!, :ratios]], label = string(s2), t = :steppre)
    xaxis!("logratio")
    plot_name = string(feature) * " & " * string(metric) * " pp"
    title!(plot_name)

    png(plot_name)

    return
end

for feature in [:stepper, :use_curve_search, :use_corr, :shift], metric in [:solve_time, :iters]
    perf_prof(feature = feature, metric = metric)
end

# anything else that can get plotted in latex
function make_csv()
    all_df = CSV.read(bench_file, DataFrame)
    select!(all_df, :use_corr, :stepper, :solve_time, :iters)
    CSV.write(joinpath(output_folder, "df_long.csv"), df_agg)
    return
end


# comb_df = filter(:stepper => isequal("Hypatia.Solvers.CombinedStepper{Float64}"), dropmissing(all_df))
# pc_df = filter(t -> t.stepper == "Hypatia.Solvers.PredOrCentStepper{Float64}" && t.use_curve_search == true, dropmissing(all_df))
# comb_times = (comb_df[!, :solve_time])
# pc_times = (pc_df[!, :solve_time])
#
# plot(comb_times, seriestype=:stephist)
# plot!(pc_times, seriestype=:stephist)

# missig instances
# all_df = CSV.read(bench_file, DataFrame)
# insts1 = filter(t -> t.stepper == "Hypatia.Solvers.PredOrCentStepper{Float64}" && t.use_curve_search == true, dropmissing(all_df))
# insts2 = filter(t -> t.stepper == "Hypatia.Solvers.CombinedStepper{Float64}" && t.shift == 1, dropmissing(all_df))
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

function is_nonsymm(cones, extender)
    if extender != "nothing"
        return false
    else
        for c in cones
            if !(c in ["Nonnegative", "EpiNormEucl", "PosSemidefTri"])
                return true
            end
        end
        return false
    end
end

all_df[!, :is_nonsymm] = is_nonsymm.(all_df[!, :cone_types], all_df[!, :extender])

histogram(log10.(all_df[!, :solve_time]))
title!("log10 solve time")
png("solvehist")

histogram(log10.(sum(eachcol(all_df[!, [:n, :p, :q]]))))
title!("log10 n + p + q")
png("npqhist")

histogram(log10.(max.(0.01, all_df[!, :n] - all_df[!, :p])))
title!("log10(n - p)")
png("nphist")
