using CSV
using DataFrames
using Plots

bench_file = joinpath("bench2", "bench.csv")
output_folder = mkpath(joinpath(@__DIR__, "results"))

shifted_geomean(x; shift = 0) = exp(sum(log, x .- shift) / length(x))

# aggregate stuff
function post_process()
    all_df = CSV.read(bench_file, DataFrame)
    select!(all_df, :stepper, :use_corr, :use_curve_search, :solve_time, :iters, :status)

    df_agg = combine(groupby(all_df, [:stepper, :use_corr, :use_curve_search]),
        :solve_time => shifted_geomean => :solve_time_geomean,
        :iters => shifted_geomean => :iters_geomean,
        :status => (x -> count(isequal("Optimal"), x)) => :converged,
        )
    CSV.write(joinpath(output_folder, "df_agg.csv"), df_agg)

    return
end
post_process()

# performance profiles
function perf_prof()
    feature = :use_corr
    metric = :solve_time
    # s1 = "TRUE"
    # s2 = "FALSE"
    s1 = true
    s2 = false

    all_df = CSV.read(bench_file, DataFrame)
    select!(all_df,
        :inst_data,
        feature,
        :solve_time => ByRow(x -> (isnan(x) ? 1800 : min(x, 1800))) => :stime,
        )
    transform!(all_df, :stime => (x -> x ./ minimum(x)) => :ratios)
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
