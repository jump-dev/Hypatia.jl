#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
process predefined datasets into txt files for fast importing via readdlm
=#

import DataFrames
import CSV
import DelimitedFiles

# iris dataset
function get_iris_data()
    df = CSV.read(joinpath(@__DIR__, "iris_raw.csv"))
    # only use setosa species
    dfsub = df[
        df.species .== "setosa",
        [:sepal_length, :sepal_width, :petal_length, :petal_width],
    ] # n = 4
    X = convert(Matrix{Float64}, dfsub)
    return X
end

open(joinpath(@__DIR__, "iris.txt"), "w") do io
    return DelimitedFiles.writedlm(io, get_iris_data())
end

# lung cancer dataset from https://github.com/therneau/survival (cancer.rda)
# description at https://github.com/therneau/survival/blob/master/man/lung.Rd
function get_cancer_data()
    df = CSV.read(
        joinpath(@__DIR__, "cancer_raw.csv"),
        missingstring = "NA",
        copycols = true,
    )
    DataFrames.dropmissing!(df, disallowmissing = true)
    # only use males with status 2
    dfsub = df[df.status .== 2, :]
    dfsub = dfsub[
        dfsub.sex .== 1,
        [:time, :age, :ph_ecog, :ph_karno, :pat_karno, :meal_cal, :wt_loss],
    ] # n = 7
    X = convert(Matrix{Float64}, dfsub)
    return X
end

open(joinpath(@__DIR__, "cancer.txt"), "w") do io
    return DelimitedFiles.writedlm(io, get_cancer_data())
end
