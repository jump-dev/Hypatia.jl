using CSV
using DataFrames

# df_all = DataFrame(CSV.File(joinpath(homedir(), "bench", "bench.csv")))
df_all = DataFrame(CSV.File(joinpath(@__DIR__, "bench.csv")))

params_map = Dict(
    "DensityEstJuMP"    => ([:m, :twod], [2, 3]), # TODO 2d?
    "ExpDesignJuMP"     => ([:logdetobj, :k], [5, 1]),
    "MatrixCompletionJuMP" => ([:d1, :d2], [1, 2]),
    "MatrixRegressionJuMP" => ([:m], [2]),
    "NearestPSDJuMP"    => ([:compl, :d], [2, 1]),
    "PolyMinJuMP"       => ([:m, :twod], [1, 2]),
    "PortfolioJuMP"     => ([:k], [1]),
    "ShapeConRegrJuMP"  => ([:m, :twod], [1, 5]),
    )

# TODO refine these - distinguish dying on model building vs solve/check
status_map = Dict(
    "Optimal" => "co",
    "KilledTime" => "tl",
    "TimeLimit" => "tl",
    "SlowProgress" => "sp",
    "NumericalFailure" => "er",
    # "KilledMemory" => "rl",
    "SkippedSolveCheck" => "sk",
    "SetupModelKilledTime" => "tl",
    "SolveCheckKilledTime" => "tl",
    "SetupModelKilledMemory" => "rl",
    "SolveCheckKilledMemory" => "rl",
    # TODO remove if not needed
    "SetupModelCaughtError" => "er",
    "SolveCheckCaughtError" => "er",
    )

rel_tol_satisfied(a, b) = (abs(a - b) / (1 + max(abs(a), abs(b))) < 1e-5)

transform!(
    df_all,
    [:inst_set, :solver] => ((x, y) -> x .* "_" .* y) => :inst_solver,
    [:x_viol, :y_viol, :z_viol, :rel_obj_diff] => ByRow((res...) -> (all(isfinite.(res)) && maximum(res) < 1e-6)) => :converged,
    :status => ByRow(x -> status_map[x]) => :status,
    )

# makes wide DataFrame
for edf in groupby(df_all, :example)
    ex_name = edf.example[1]
    @info("starting $ex_name")
    ex_df = copy(edf)
    (inst_keys, positions) = params_map[ex_name]
    # make columns out of tuple
    for (name, position) in zip(inst_keys, positions)
        transform!(ex_df, :inst_data => ByRow(x -> eval(Meta.parse(x))[position]) => name)
    end

    # checks objectives if solver claims optimality
    for subdf in groupby(ex_df, inst_keys)
        # check all pairs of converged results
        co_idxs = findall(subdf[:status] .== "co")
        if length(co_idxs) >= 2
            for i in eachindex(co_idxs)
                first_optval = subdf[co_idxs[i], :prim_obj]
                other_optvals = subdf[co_idxs[Not(i)], :prim_obj]
                if !all(rel_tol_satisfied.(other_optvals, first_optval))
                    println("objective values of: $(ex_name) $(subdf[:inst_data][1]) do not agree")
                end
            end
        end
    end

    unstacked_dims = [
        unstack(ex_df, inst_keys, :inst_set, v, renamecols = x -> Symbol(v, :_, x))
        for v in [:n, :p, :q]
        ]
    unstacked_res = [
        unstack(ex_df, inst_keys, :inst_solver, v, renamecols = x -> Symbol(v, :_, x))
        for v in [:status, :converged, :iters, :solve_time]
        ]
    ex_df_wide = join(unstacked_dims..., unstacked_res..., on = inst_keys)

    CSV.write(joinpath(@__DIR__, ex_name * "_wide.csv"), ex_df_wide)
end
;
