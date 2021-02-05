using CSV
using DataFrames
using Plots

MAX_TIME = 1800
MAX_ITER = 250

bench_file = joinpath("bench2", "nat", "bench_nat_all.csv")
output_folder = mkpath(joinpath(@__DIR__, "results"))

shifted_geomean_notmissing(x; shift = 0) = exp(sum(log, skipmissing(x) .+ shift) / count(!ismissing, x))
shifted_geomean_all(x; shift = 0, cap = Inf) = exp(sum(log, coalesce.(x, cap) .+ shift) / length(x))
function shifted_geomean_opt(metric, status; shift = 0)
    idxs = .!(ismissing.(status)) .& (status .== "Optimal")
    return exp(sum(log, metric[idxs] .+ shift) / length(idxs))
end
shifted_geomean_all_opt(metric, all_opt; shift = 0) = exp(sum(log, metric[all_opt] .+ shift) / count(all_opt))

# comb_df = filter(:stepper => isequal("Hypatia.Solvers.CombinedStepper{Float64}"), dropmissing(all_df))
# pc_df = filter(t -> t.stepper == "Hypatia.Solvers.PredOrCentStepper{Float64}" && t.use_curve_search == true, dropmissing(all_df))
# comb_times = (comb_df[!, :solve_time])
# pc_times = (pc_df[!, :solve_time])
#
# plot(comb_times, seriestype=:stephist)
# plot!(pc_times, seriestype=:stephist)

# aggregate stuff
function post_process()
    all_df = CSV.read(bench_file, DataFrame)
    all_df = combine(groupby(all_df, :inst_data), names(all_df), :status => (x -> all(isequal.("Optimal", x))) => :all_opt)
    select!(all_df, :stepper, :use_corr, :use_curve_search, :solve_time, :iters, :status, :all_opt)

    df_agg = combine(groupby(all_df, [:stepper, :use_corr, :use_curve_search]),
        :solve_time => shifted_geomean_notmissing => :solve_time_geomean_notmissing,
        :iters => shifted_geomean_notmissing => :iters_geomean_notmissing,
        [:solve_time, :status] => shifted_geomean_opt => :solve_time_geomean_this_opt,
        [:iters, :status] => shifted_geomean_opt => :iters_geomean_this_opt,
        [:solve_time, :all_opt] => shifted_geomean_all_opt => :solve_time_geomean_all_opt,
        [:iters, :all_opt] => shifted_geomean_all_opt => :iters_geomean_all_opt,
        :solve_time => (x -> shifted_geomean_all(x, cap = MAX_TIME)) => :solve_time_geomean_all,
        :iters => (x -> shifted_geomean_all(x, cap = MAX_ITER)) => :iters_geomean_all,
        :status => (x -> count(isequal("Optimal"), x)) => :optimal,
        :status => (x -> count(isequal("NumericalFailure"), x)) => :numerical,
        :status => (x -> count(isequal("SlowProgress"), x)) => :slowprogress,
        :status => (x -> count(startswith("SolveCheckKilled"), x)) => :killed,
        :status => (x -> count(startswith("Setup"), x)) => :setup,
        :status => (x -> count(isequal("TimeLimit"), x)) => :timelimit,
        :status => (x -> count(isequal("IterationLimit"), x)) => :iterationlimit,
        :status => (x -> count(startswith("Skipped"), x)) => :skip,
        :status => length => :total,
        )
    CSV.write(joinpath(output_folder, "df_agg_nat.csv"), df_agg)

    return
end
post_process()

# performance profiles, currently hardcoded for corrector vs no corrector
function perf_prof()
    feature = :use_corr
    metric = :iters
    # metric = :iters
    # s1 = "TRUE"
    # s2 = "FALSE"
    s1 = true
    s2 = false

    all_df = CSV.read(bench_file, DataFrame)
    filter!(t -> t.stepper == "Hypatia.Solvers.PredOrCentStepper{Float64}" && t.use_curve_search == false,
        all_df
        )
    select!(all_df,
        feature,
        :solve_time => ByRow(x -> (ismissing(x) ? MAX_TIME : min(x, MAX_TIME))) => :solve_time,
        :iters => ByRow(x -> (ismissing(x) ? MAX_TIME : min(x, MAX_ITER))) => :iters,
        )
    transform!(all_df, metric => (x -> x ./ minimum(x)) => :ratios)
    sort!(all_df, :ratios)

    nsolvers = 2
    npts = nrow(all_df)

    # for metric in [:solve_time, :iters]
    plot(xlim = (0, log10(maximum(all_df[!, :ratios])) + 1), ylim = (0, 1))
    subdf = all_df[all_df[!, feature] .== s1, :]
    plot!(log10.(all_df[!, :ratios]), [sum(subdf[!, :ratios] .<= ti) ./ npts * nsolvers for ti in all_df[!, :ratios]], label = s1, t = :steppre)

    subdf = all_df[all_df[!, feature] .== s2, :]
    plot!(log10.(all_df[!, :ratios]), [sum(subdf[!, :ratios] .<= ti) ./ npts * nsolvers for ti in all_df[!, :ratios]], label = s1, t = :steppre)
    xaxis!("logratio")
    title!(string(feature) * " / " * string(metric) * " pp")

    return
end
perf_prof()

# anything else that can get plotted in latex
function make_csv()
    all_df = CSV.read(bench_file, DataFrame)
    select!(all_df, :use_corr, :stepper, :solve_time, :iters)
    CSV.write(joinpath(output_folder, "df_long.csv"), df_agg)
    return
end
