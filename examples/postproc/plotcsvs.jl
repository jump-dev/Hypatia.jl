# function make_plot_csv(ex)
#     df = CSV.read(ex * "_wide.csv")
#     if ex == "MatrixCompletionJuMP"
#         transform!(df, [:d1, :d2] => ((x, y) -> round.(Int, y ./ x)) => :gr)
#         x_var = :d1
#     elseif ex in ["DensityEstJuMP", "PolyMinJuMP", "ShapeConRegrJuMP"]
#         rename!(df, :m => :gr)
#         x_var = :d
#     else
#         ex_short = ex[1:(match(r"JuMP", ex).offset + 3)] # in case of suffix
#         x_var = params_map[ex_short].inst_keys[1]
#         df.gr = fill("", nrow(df))
#     end
#     transform!(df, [:converged_nat_Hypatia, :solve_time_nat_Hypatia] => ((x, y) -> (!ismissing(x) && x ? y : missing)) => :nat)
#     transform!(df, [:converged_ext_Hypatia, :solve_time_ext_Hypatia] => ((x, y) -> (!ismissing(x) && x ? y : missing)) => :ext)
#     transform!(df, [:converged_ext_Mosek, :solve_time_ext_Mosek] => ((x, y) -> (!ismissing(x) && x ? y : missing)) => :mosek)
#
#     success_df = select(df, x_var, :gr, :nat, :ext, :mosek)
#     for subdf in groupby(success_df, :gr)
#         v = subdf.gr[1]
#         CSV.write(ex * "$(v)_plot.csv", select(subdf, Not(:gr)))
#     end
#     return
# end

function make_plot_csv(ex)
    df = CSV.read(ex * "_wide.csv")
    # only plot converged instances
    success_df = DataFrame(x = Int[], gr = Union{Int, String}[], nat = Union{Missing, Float64}[], ext = Union{Missing, Float64}[], mosek = Union{Missing, Float64}[])
    for r in eachrow(df)
        nat_time = (coalesce(r.converged_nat_Hypatia, false) ? r.solve_time_nat_Hypatia : missing)
        ext_time = (coalesce(r.converged_ext_Hypatia, false) ? r.solve_time_ext_Hypatia : missing)
        mosek_time = (coalesce(r.converged_ext_Mosek, false) ? r.solve_time_ext_Mosek : missing)
        if ex == "MatrixCompletionJuMP"
            push!(success_df, (r.d1, round(Int, r.d2 / r.d1), nat_time, ext_time, mosek_time))
        elseif ex in ["DensityEstJuMP", "PolyMinJuMP", "ShapeConRegrJuMP"]
            push!(success_df, (r.d, r.m, nat_time, ext_time, mosek_time))
        else
            ex_short = ex[1:(match(r"JuMP", ex).offset + 3)] # in case of suffix
            x_var = params_map[ex_short].inst_keys[1]
            push!(success_df, (r[x_var], "", nat_time, ext_time, mosek_time))
        end
    end
    for subdf in groupby(success_df, :gr)
        v = subdf.gr[1]
        CSV.write(ex * "$(v)_plot.csv", select(subdf, Not(:gr)))
    end
    return
end

make_plot_csv.([
    # "DensityEstJuMP",
    "ExpDesignJuMP_logdetobj_true",
    "ExpDesignJuMP_logdetobj_false",
    "MatrixCompletionJuMP",
    "PortfolioJuMP",
    ])
