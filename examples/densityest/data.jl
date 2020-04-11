#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

list of predefined density functions
=#

# support on [-1, 1]^n, so won't get rescaled in build
# assume independence between dimensions, cumulative_inv always univariate
function densityest_data(density_name::Symbol, T::Type{<:Real} = Float64)
    if density_name == :density1
        density = (x -> T(0.5))
        cumulative_inv = (x -> (x - T(0.5)) * 2)
        num_obs = 400
        n = 1
        deg = 2
    elseif density_name == :density2
        density = ((x, y) -> T(0.25))
        cumulative_inv = (x -> (x - T(0.5)) * 2)
        num_obs = 400
        n = 2
        deg = 2
    elseif density_name == :density3
        density = (x -> T(0.5) - x / 2)
        cumulative_inv = (x -> 1 - sqrt(-4 * (x - 1)))
        num_obs = 400
        n = 1
        deg = 2
    else
        error("unknown name $(density_name)")
    end
    return (density, cumulative_inv, num_obs, n, deg)
end
