using DataFrames
using DataFramesMeta
using CSV

for i in [1, 3, 5]
    longres = CSV.read(joinpath(@__DIR__(), "synthetic$(i).csv"), comment="#")
    categorical!(longres, :signal_ratio)
    longres[:use_wsos] = string.(longres[:use_wsos])
    longres[:use_wsos][longres[:use_wsos] .== "true"] .= "wsos"
    longres[:use_wsos][longres[:use_wsos] .== "false"] .= "sdp"
    longres_tm = longres[[:signal_ratio, :deg, :use_wsos, :tm]]
    # wideres = unstack(longres_tm, :use_wsos, :tm)
    unstack(longres_tm, :use_wsos, :tm) |> CSV.write("syn$(i).csv")
    smry_res = aggregate(wideres, [:deg, :signal_ratio], [mean, minimum, maximum]) |> CSV.write("syn$(i).csv")
end
