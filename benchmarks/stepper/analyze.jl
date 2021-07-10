#=
analyze stepper benchmark results
see stepper/README.md
=#

using CSV
using DataFrames
using Printf
import BenchmarkProfiles

keep_set = "various"

enhancements = [
    "basic",
    "prox",
    "TOA",
    "curve",
    "comb",
    ]

compare_pairs = [
    ["basic", "prox"],
    ["prox", "toa"],
    ["toa", "curve"],
    ["curve", "comb"],
    ["basic", "comb"],
    ]

# geomean shifts
time_shift = 1e-3
total_shift = 1e-4
piter_shift = 1e-5

# file locations
bench_file = joinpath(@__DIR__, "raw", "bench.csv")
output_dir = mkpath(joinpath(@__DIR__, "analysis"))
tex_dir = mkpath(joinpath(output_dir, "tex"))
stats_dir = mkpath(joinpath(output_dir, "stats"))
csv_dir = mkpath(joinpath(output_dir, "csvs"))

# get extra info about runs; uses hardcoded enhancement names
function extra_stats(all_df)
    all_df = transform(all_df, [:n, :p, :q] => ((x, y, z) -> x .+ y .+ z) => :npq)

    # basic or comb converged
    for enh in ("basic", "comb")
        enh_conv = filter(t -> ((t.enhancement == enh) && t.conv), all_df)
        enh_data = select(enh_conv,
            :npq, :iters, :solve_time,
            :iters => ByRow(log10) => :log_iters,
            :solve_time => ByRow(log10) => :log_solve_time,
            [:time_uprhs, :solve_time] => ((x, y) -> x ./ y) => :prop_rhs,
            )
        CSV.write(joinpath(csv_dir, enh * "conv.csv"), enh_data)
    end

    # basic/comb both converged
    two_solver = filter(t -> (t.enhancement in ("basic", "comb")), all_df)
    two_solver = combine(groupby(two_solver, :inst_key), names(all_df),
        :conv => all => :two_conv)
    two_solver_conv = filter(t -> t.two_conv, two_solver)
    rel_impr = (x -> (x[1] - x[2]) / x[1])
    two_solver_conv = combine(groupby(two_solver_conv, :inst_key),
        [:enhancement, :solve_time], :solve_time => rel_impr => :time_impr,
        [:enhancement, :iters], :iters => rel_impr => :iters_impr,
        )
    filter!(t -> (t.enhancement == "comb"), two_solver_conv)
    CSV.write(joinpath(csv_dir, "basiccombconv.csv"),
        select(two_solver_conv, :solve_time, :time_impr, :iters, :iters_impr))

    # logged stats for instances
    basic_df = filter(t -> (t.enhancement == "basic"), all_df)
    CSV.write(joinpath(csv_dir, "inst_stats.csv"), select(basic_df,
        :num_cones => ByRow(log10) => :log_numcones,
        :npq => ByRow(log10) => :log_npq,
        ),)

    # update examplestats.csv with unique cones and instance count for each example
    exstats = open(joinpath(stats_dir, "examplestats.csv"), "w")
    println(exstats, "example,num_instances,cones")
    for g in groupby(basic_df, :example, sort = true)
        ex = first(g).example
        n = nrow(g)
        cone_lists = [eval(Meta.parse.(x)) for x in g[!, :cone_types]]
        cones = unique(vcat(cone_lists...))
        println(exstats, "$ex,$n,$cones")
    end
    close(exstats)

    # count instances with non-default solver options
    nondef = filter!(!ismissing, basic_df[!, :nondefaults])
    count_nondefault = length(nondef)
    @info("$count_nondefault instances have non-default solver options")
    if !iszero(count_nondefault)
        @info("unique non-default solver options are:")
        for nd in unique(nondef)
            println(nd)
        end
    end

    # count number of instances in "every"
    count_every = nrow(unique(select(filter(t -> t.every_conv, all_df),
        :inst_key)))
    @info("$count_every instances are in the every set")

    return
end

# generic functions without hardcoded enhancement names

function post_process()
    all_df = preprocess_df()
    if isempty(all_df)
        error("input CSV has no $keep_set set instances with a full list of runs")
    end

    make_agg_tables(all_df)
    make_subtime_tables(all_df)

    for comp in compare_pairs, metric in [:solve_time, :iters]
        make_perf_profiles(all_df, comp, metric)
    end
    extra_stats(all_df)

    return
end

process_entry(x::Float64) = @sprintf("%.2f", x)

process_entry(x::Int) = string(x)

get_cap(x::AbstractVector, y::AbstractVector) = 2 * maximum(x[y])

function shifted_geomean(
    metric::AbstractVector{<:Real},
    conv::AbstractVector{Bool};
    shift::Real = 0,
    cap::AbstractVector = fill(Inf, length(conv)),
    use_cap::Bool = false,
    )
    if use_cap
        @assert length(cap) == length(metric)
        x = copy(metric)
        x[.!conv] .= cap[.!conv]
    else
        x = metric[conv]
    end
    @assert all(isfinite, x)
    return exp(sum(log, x .+ shift) / length(x)) - shift
end

# get enhancement name from solver options
function get_enhancement(x)
    sol_opt = eval(Meta.parse(x))
    @assert length(sol_opt) == 1
    return sol_opt[1]
end

function preprocess_df()
    all_df = CSV.read(bench_file, DataFrame)

    # only keep wanted instance set
    filter!(t -> (t.inst_set == keep_set), all_df)

    # get enhancement name from solver options
    transform!(all_df, :solver_options => ByRow(get_enhancement) => :enhancement)

    # check if any instances could be duplicates:
    possible_dupes = nonunique(all_df, [:enhancement, :example, :inst_data,
        :n, :p, :q, :nu, :cone_types, :num_cones, :max_q])
    if any(possible_dupes)
        df_dupes = unique!(all_df[possible_dupes, [:example, :model_type,
            :inst_set, :inst_num, :inst_data, :extender]])
        println()
        @warn("possible instance/option duplicates detected; inspect instance set " *
            "for duplicates of each of the below (unique) rows:")
        println("\n", df_dupes, "\n")
    end

    # get converged instances, identify by instance key
    ok_status = ["Optimal", "PrimalInfeasible", "DualInfeasible"]
    str_missing(s) = (ismissing(s) ? "" : string(s))
    transform!(all_df,
        :status => ByRow(x -> (!ismissing(x) && x in ok_status)) => :conv,
        # identify instances by example + model_type + inst_data + extender
        [:example, :model_type, :inst_data, :extender] =>
        ((s1, s2, s3, s4) -> s1 .* s2 .* s3 .* str_missing.(s4)) => :inst_key,
        )

    # assumes that nothing returned incorrect status, which is checked manually
    all_df = combine(groupby(all_df, :inst_key), names(all_df),
        :conv => all => :every_conv)

    # assert all instances are solved by at least one stepper (removes noise)
    none_df = combine(groupby(all_df, :inst_key), :inst_key,
        :conv => (x -> !any(x)) => :none_conv)
    @assert isempty(filter(t -> t.none_conv, none_df))

    # assert all instances took at least one iteration
    @assert minimum(all_df[!, :iters]) >= 1

    return all_df
end

function make_agg_tables(all_df)
    # calculate caps for replacing time/iters of unconverged instances
    # use double the largest value for the same instance across all steppers
    # assumes instances are in the same order for all steppers
    max_df = combine(groupby(all_df, :inst_key),
        [:solve_time, :conv] => get_cap => :max_time,
        [:iters, :conv] => get_cap => :max_iters,
        )
    cap(x::Symbol) = max_df[!, x]

    # collect aggregated summary statistics
    df_agg = combine(groupby(all_df, :enhancement),
        # geometric mean over instances where every stepper converged
        [:solve_time, :every_conv] => ((x, y) ->
            shifted_geomean(x, y, shift = time_shift)) => :time_geomean_everyconv,
        [:iters, :every_conv] => ((x, y) ->
            shifted_geomean(x, y, shift = 1)) => :iters_geomean_everyconv,
        # geometric mean over instances where this stepper converged
        [:solve_time, :conv] => ((x, y) ->
            shifted_geomean(x, y, shift = time_shift)) => :time_geomean_thisconv,
        [:iters, :conv] => ((x, y) ->
            shifted_geomean(x, y, shift = 1)) => :iters_geomean_thisconv,
        # geometric mean over all instances, use caps on unconverged isntances
        [:solve_time, :conv] => ((x, y) ->
            shifted_geomean(x, y, cap = cap(:max_time), use_cap = true,
            shift = time_shift)) => :time_geomean_all,
        [:iters, :conv] => ((x, y) ->
            shifted_geomean(x, y, shift = 1, cap = cap(:max_iters),
            use_cap = true)) => :iters_geomean_all,
        :status => (x -> count(isequal("Optimal"), x)) => :optimal,
        :status => (x -> count(isequal("PrimalInfeasible"), x)) => :priminfeas,
        :status => (x -> count(isequal("DualInfeasible"), x)) => :dualinfeas,
        :status => (x -> count(isequal("NumericalFailure"), x)) => :numerical,
        :status => (x -> count(isequal("SlowProgress"), x)) => :slowprogress,
        :status => (x -> count(isequal("TimeLimit"), x)) => :timelimit,
        :status => (x -> count(isequal("IterationLimit"), x)) => :iterationlimit,
        :status => length => :total,
        )

    sort!(df_agg, order(:enhancement, by = (x ->
        findfirst(isequal(x), lowercase.(enhancements)))))
    CSV.write(joinpath(stats_dir, "agg.csv"), df_agg)

    # prepare latex table

    # combine feasible and infeasible statuses
    transform!(df_agg, [:optimal, :priminfeas, :dualinfeas] =>
        ByRow((x...) -> sum(x)) => :converged)

    cols = [:converged, :iters_geomean_everyconv, :iters_geomean_thisconv,
        :iters_geomean_all, :time_geomean_everyconv, :time_geomean_thisconv,
        :time_geomean_all]
    sep = " & "
    tex = open(joinpath(tex_dir, "agg.tex"), "w")

    for i in 1:length(enhancements)
        row_str = enhancements[i]
        for c in cols
            subdf = df_agg[!, c]
            x = (startswith(string(c), "time") ? subdf[i] * 1000 : subdf[i])
            row_str *= sep * process_entry(x)
        end
        row_str *= " \\\\"
        println(tex, row_str)
    end
    close(tex)

    return
end

function make_subtime_tables(all_df)
    divfunc(x, y) = (x ./ y)
    init_cols = [:time_rescale, :time_initx, :time_inity, :time_unproc,
        :time_loadsys]
    transform!(all_df,
        [:time_upsys, :iters] => divfunc => :time_upsys_piter,
        [:time_uprhs, :iters] => divfunc => :time_uprhs_piter,
        [:time_getdir, :iters] => divfunc => :time_getdir_piter,
        [:time_search, :iters] => divfunc => :time_search_piter,
        init_cols => ((x...) -> sum(x)) => :time_init,
        )

    metrics = [:init, :lhs, :rhs, :direc, :search, :lhs_piter,
        :rhs_piter, :direc_piter, :search_piter]
    sets = [:_thisconv, :_everyconv, :_all]

    # calculate caps for replacing time/iters of unconverged instances
    # use double the largest value for the same instance across all steppers
    # assumes instances are in the same order for all steppers
    max_df = combine(groupby(all_df, :inst_key),
        [:time_init, :conv] => get_cap => :max_init,
        [:time_upsys, :conv] => get_cap => :max_upsys,
        [:time_uprhs, :conv] => get_cap => :max_uprhs,
        [:time_getdir, :conv] => get_cap => :max_getdir,
        [:time_search, :conv] => get_cap => :max_search,
        [:time_upsys_piter, :conv] => get_cap => :max_upsys_iter,
        [:time_uprhs_piter, :conv] => get_cap => :max_uprhs_iter,
        [:time_getdir_piter, :conv] => get_cap => :max_getdir_iter,
        [:time_search_piter, :conv] => get_cap => :max_search_iter,
        )
    cap(x::Symbol) = max_df[!, x]

    function get_subtime_df(set, convcol, use_cap)
        subtime_df = combine(groupby(all_df, :enhancement),
            [:time_init, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = cap(:max_init),
                use_cap = use_cap)) => Symbol(:init, set),
            [:time_upsys, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = cap(:max_upsys),
                use_cap = use_cap)) => Symbol(:lhs, set),
            [:time_uprhs, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = cap(:max_uprhs),
                use_cap = use_cap)) => Symbol(:rhs, set),
            [:time_getdir, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = cap(:max_getdir),
                use_cap = use_cap)) => Symbol(:direc, set),
            [:time_search, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = cap(:max_search),
                use_cap = use_cap)) => Symbol(:search, set),
            [:time_upsys_piter, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = piter_shift,
                cap = cap(:max_upsys_iter), use_cap = use_cap)) =>
                Symbol(:lhs_piter, set),
            [:time_uprhs_piter, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = piter_shift,
                cap = cap(:max_uprhs_iter), use_cap = use_cap)) =>
                Symbol(:rhs_piter, set),
            [:time_getdir_piter, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = piter_shift,
                cap = cap(:max_getdir_iter), use_cap = use_cap)) =>
                Symbol(:direc_piter, set),
            [:time_search_piter, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = piter_shift,
                cap = cap(:max_search_iter), use_cap = use_cap)) =>
                Symbol(:search_piter, set),
            )

        sort!(subtime_df, order(:enhancement,
            by = (x -> findfirst(isequal(x), lowercase.(enhancements)))))
        CSV.write(joinpath(stats_dir, "subtime" * string(set) * ".csv"),
            subtime_df)

        return subtime_df
    end

    # prepare latex table

    sep = " & "
    for s in sets
        if s == :_thisconv
            convcol = :conv
            use_cap = false
        elseif s == :_everyconv
            convcol = :every_conv
            use_cap = false
        elseif s == :_all
            convcol = :conv
            use_cap = true
        end
        subtime_df = get_subtime_df(s, convcol, use_cap)

        subtime_tex = open(joinpath(tex_dir, "subtime" * string(s) * ".tex"), "w")
        for i in 1:nrow(subtime_df)
            row_str = sep * enhancements[i]
            for m in metrics
                col = Symbol(m, s)
                row_str *= sep * process_entry(subtime_df[i, col] * 1000)
            end
            row_str *= " \\\\"
            println(subtime_tex, row_str)
        end
        close(subtime_tex)
    end

    return
end

function make_perf_profiles(all_df, comp, metric)
    pp = filter(t -> t.enhancement in comp, all_df)

    # BenchmarkProfiles.jl expects NaNs for failures
    select!(pp,
        :inst_key,
        :enhancement,
        [metric, :conv] => ByRow((x, y) -> (y ? x : NaN)) => metric,
        )
    wide_df = unstack(pp, :enhancement, metric)

    (x_plot, y_plot, max_ratio) = BenchmarkProfiles.performance_profile_data(
        Matrix{Float64}(wide_df[!, string.(comp)]), logscale = true)

    # make steps like :steppost in Plots
    for s in 1:2
        x = vcat(0, repeat(x_plot[s], inner = 2))
        y = vcat(0, 0, repeat(y_plot[s][1:(end - 1)], inner = 2), y_plot[s][end])
        CSV.write(joinpath(csv_dir, comp[s] * "_vs_" * comp[2 - s + 1] * "_" *
            string(metric) * ".csv"), DataFrame(x = x, y = y))
    end

    return
end

post_process()
;
