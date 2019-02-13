#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using DataFrames
using CSV
include(joinpath(@__DIR__(), "jump.jl"))

# iris dataset
function iris_data()
    df = CSV.read(joinpath(@__DIR__, "data", "iris.csv"))
    dropmissing!(df, disallowmissing = true)
    # only use setosa species
    xcols = [:sepal_length, :sepal_width, :petal_length, :petal_width]
    dfsub = df[df.species .== "setosa", xcols]
    X = convert(Matrix{Float64}, dfsub)
    X .-= 0.5 * (minimum(X, dims=1) + maximum(X, dims=1))
    X ./= (0.5 * (maximum(X, dims=1) - minimum(X, dims=1)))
    n = 4
    return (X, n)
end

# lung cancer dataset from https://github.com/therneau/survival (cancer.rda)
function cancer_data()
    df = CSV.read(joinpath(@__DIR__, "data", "cancer.csv"), missingstring = "NA")
    dropmissing!(df, disallowmissing = true)
    # only use males with status 2
    dfsub = df[df.status .== 2, :]
    dfsub = dfsub[dfsub.sex .== 1, [:time, :age, :ph_ecog, :ph_karno, :pat_karno, :meal_cal, :wt_loss]]
    X = convert(Matrix{Float64}, dfsub)
    X .-= 0.5 * (minimum(X, dims=1) + maximum(X, dims=1))
    X ./= (0.5 * (maximum(X, dims=1) - minimum(X, dims=1)))
    n = 7
    return (X, n)
end

function run_hard_densityest()
    degrees = 4:2:6

    datasets = [
        iris_data,
        cancer_data,
        ]

    for d in degrees, s in datasets
        model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer,
            use_dense = true,
            verbose = true,
            system_solver = SO.QRCholCombinedHSDSystemSolver,
            linear_model = MO.PreprocessedLinearModel,
            max_iters = 250,
            time_limit = 3.6e3,
            tol_rel_opt = 1e-5,
            tol_abs_opt = 1e-6,
            tol_feas = 1e-6,
            ))

        println()
        @show d
        @show s
        println()

        (X, n) = s()
        dom = MU.Box(-ones(n), ones(n))
        build_JuMP_densityest(model, X, d, dom)

        (val, runtime, bytes, gctime, memallocs) = @timed JuMP.optimize!(model)

        println()
        @show runtime
        @show bytes
        @show gctime
        @show memallocs
        println("\n\n")
    end
end

run_hard_densityest()
