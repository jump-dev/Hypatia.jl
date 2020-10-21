function make_plot_csv(ex)
    df = CSV.read(ex * "_wide.csv")
    # only plot converged isntances
    success_df = DataFrame(x = Int[], gr = Union{Int, String}[], nat = Union{Missing, Float64}[], ext = Union{Missing, Float64}[], mosek = Union{Missing, Float64}[])
    if ex == "MatrixCompletionJuMP"
        group_vals = [5, 10]
    elseif ex in ["DensityEstJuMP", "PolyMinJuMP", "ShapeConRegrJuMP"]
        group_vals = [2, 3, 4]
    end
    for r in eachrow(df)
        nat_time = (coalesce(r.converged_nat_Hypatia, false) ? r[:solve_time_nat_Hypatia] : missing)
        ext_time = (coalesce(r.converged_ext_Hypatia, false) ? r[:solve_time_ext_Hypatia] : missing)
        mosek_time = (coalesce(r.converged_ext_Mosek, false) ? r[:solve_time_ext_Mosek] : missing)
        if ex == "MatrixCompletionJuMP"
            push!(success_df, (r[:d1], round(Int, r[:d2] / r[:d1]), nat_time, ext_time, mosek_time))
        elseif ex in ["DensityEstJuMP", "PolyMinJuMP", "ShapeConRegrJuMP"]
            push!(success_df, (r[:d], r[:m], nat_time, ext_time, mosek_time))
        else
            x_var = params_map[ex].inst_keys[1]
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
    "MatrixCompletionJuMP",
    "PortfolioJuMP",
    ])
