#=
process predefined datasets into txt files for fast importing via readdlm
=#

using LinearAlgebra
import DataFrames
import CSV
import DelimitedFiles

# naics5811 dataset (save as naics5811.csv)
# see https://arxiv.org/pdf/1509.08165v1.pdf (example 3)
# data obtained from http://www.nber.org/data/nbprod2005.html
function get_naics5811_data()
    df = CSV.read(joinpath(@__DIR__, "naics5811.csv"), copycols = true)
    DataFrames.deleterows!(df, 157) # outlier
    # number of non production employees
    df[!, :prode] .= df[!, :emp] - df[!, :prode]
    # group by industry codes
    df_aggr = DataFrames.aggregate(DataFrames.dropmissing(df), :naics, sum)

    # four covariates: non production employees, production worker hours, production workers, total capital stock
    # use the log transform of covariates
    X = log.(convert(Matrix{Float64}, df_aggr[!, [:prode_sum, :prodh_sum, :prodw_sum, :cap_sum]])) # n = 4
    # value of shipment
    y = convert(Vector{Float64}, df_aggr[!, :vship_sum])
    # mean center
    X .-= sum(X, dims = 1) ./ size(X, 1)
    y .-= sum(y) / length(y)
    # normalize to unit norm
    X ./= norm.(eachcol(X))'
    y /= norm(y)

    return hcat(X, y) # concatenate columns of X and y
end

open(joinpath(@__DIR__, "naics5811.txt"), "w") do io
    DelimitedFiles.writedlm(io, get_naics5811_data())
end
