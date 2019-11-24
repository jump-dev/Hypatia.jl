#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhedron
second-order cone (EpiNormEucl) extended formulation inspired by MOI bridge
=#

using LinearAlgebra
import Random
using Test
import Hypatia
const CO = Hypatia.Cones

# modified from https://github.com/JuliaOpt/MathOptInterface.jl/blob/master/src/Bridges/Constraint/geomean.jl
function log_floor(n, i)
    if n <= 2 ^ i
        i
    else
        log_floor(n, i + 1)
    end
end
function log_floor(n::Integer)
    @assert n > zero(n)
    log_floor(n, zero(n))
end

function maxvolume(
    T::Type{<:Real},
    n::Int;
    use_hypogeomean::Bool = false,
    use_power::Bool = false,
    use_epinormeucl::Bool = false,
    )
    @assert use_hypogeomean + use_power + use_epinormeucl == 1
    poly_hrep = Matrix{T}(I, n, n)
    poly_hrep .+= T.(randn(n, n)) / n
    c = vcat(-1, zeros(T, n))
    A = hcat(zeros(T, n), poly_hrep)
    b = ones(T, n)

    if use_hypogeomean
        G = -Matrix{T}(I, n + 1, n + 1)
        h = zeros(T, n + 1)
        cones = CO.Cone{T}[CO.HypoGeomean{T}(fill(inv(T(n)), n))]

    elseif use_power
        cones = CO.Cone{T}[]
        # number of 3-dimensional power cones needed is n - 1, number of new variables is n - 2
        len_power = 3 * (n - 1)
        G_geo_orig = zeros(T, len_power, n)
        G_geo_newvars = zeros(T, len_power, n - 2)
        c = vcat(c, zeros(T, n - 2))
        A = hcat(A, zeros(T, n, n - 2))

        # first cone is a special case since two of the original variables participate in it
        G_geo_orig[1, 1] = -1
        G_geo_orig[2, 2] = -1
        G_geo_newvars[3, 1] = -1
        push!(cones, CO.Power{T}(fill(inv(T(2)), 2), 1))
        offset = 4
        # loop over new vars
        for i in 1:(n - 3)
            G_geo_newvars[offset + 2, i + 1] = -1
            G_geo_newvars[offset + 1, i] = -1
            G_geo_orig[offset, i + 2] = -1
            push!(cones, CO.Power{T}([inv(T(i + 2)), T(i + 1) / T(i + 2)], 1))
            offset += 3
        end
        # last row also special becuase hypograph variable is involved
        G_geo_orig[offset, n] = -1
        G_geo_newvars[offset + 1, n - 2] = -1
        G = [
            vcat(zeros(T, len_power - 1), -one(T))  G_geo_orig  G_geo_newvars
            ]
        push!(cones, CO.Power{T}([inv(T(n)), T(n - 1) / T(n)], 1))
        h = zeros(T, 3 * (n - 1))

    else
        @assert use_epinormeucl == true
        # number of variables inside geometric mean is n
        # number of layers of variables
        num_layers = log_floor(n)
        # number of new variables = 1 + 2 + ... + 2^(l - 1) = 2^l - 1
        num_new_vars = 2 ^ num_layers - 1

        c = vcat(c, zeros(T, num_new_vars))
        A = hcat(A, zeros(T, n, num_new_vars))
        rtfact = sqrt(T(2) ^ num_layers)
        # excludes original hypograph variable, padded later
        G_rsoc = zeros(T, 3 * num_new_vars, n + num_new_vars)
        cones = CO.Cone{T}[]

        offset = offset_next = 0
        row = 1
        # loop over layers, layer 1 describes hypograph variable
        for i in 1:(num_layers - 1)
            num_lvars = 2 ^ (i - 1)
            offset_next = offset + num_lvars
            # loop over variables in each layer
            for j in 1:num_lvars
                G_rsoc[row, n + offset_next + 2j - 1] = -1
                G_rsoc[row + 1, n + offset_next + 2j] = -1
                G_rsoc[row + 2, n + offset + j] = -1
                push!(cones, CO.EpiPerSquare{T}(3))
                row += 3
            end
            offset = offset_next
        end

        for j in 1:(2 ^ (num_layers - 1))
            # in the last layer, we use the original variables
            if 2j - 1 > n
                # if we are beyond the number of variables in the actual geometric mean, we are adding the buffer variable
                G_rsoc[row, n + 1] = -inv(rtfact)
            else
                G_rsoc[row, 2j - 1] = -1
            end
            if 2j > n
                # if we are beyond the number of variables in the actual geometric mean, we are adding the buffer variable
                G_rsoc[row + 1, n + 1] = -inv(rtfact)
            else
                G_rsoc[row + 1, 2j] = -1
            end
            G_rsoc[row + 2, n + offset + j] = -1
            push!(cones, CO.EpiPerSquare{T}(3))
            row += 3
        end

        # account for original hypograph variable
        G = [
            zeros(T, 3 * num_new_vars)  G_rsoc;
            one(T)  zeros(T, 1, n)  -inv(rtfact)  zeros(T, 1, num_new_vars - 1);
            zeros(T, num_new_vars, n + 1)  -Matrix{T}(I, num_new_vars, num_new_vars)
            ]
        push!(cones, CO.Nonnegative{T}(1))
        # TODO does this need to be imposed for all variables explicitly? keeping for now just in case
        push!(cones, CO.Nonnegative{T}(num_new_vars))
        h = zeros(T, 4 * num_new_vars + 1)

    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

maxvolume1(T::Type{<:Real}) = maxvolume(T, 3, use_hypogeomean = true)
maxvolume2(T::Type{<:Real}) = maxvolume(T, 3, use_power = true)
maxvolume3(T::Type{<:Real}) = maxvolume(T, 3, use_epinormeucl = true)
maxvolume4(T::Type{<:Real}) = maxvolume(T, 6, use_hypogeomean = true)
maxvolume5(T::Type{<:Real}) = maxvolume(T, 6, use_power = true)
maxvolume6(T::Type{<:Real}) = maxvolume(T, 6, use_epinormeucl = true)
maxvolume7(T::Type{<:Real}) = maxvolume(T, 25, use_hypogeomean = true)
maxvolume8(T::Type{<:Real}) = maxvolume(T, 25, use_power = true)
maxvolume9(T::Type{<:Real}) = maxvolume(T, 25, use_epinormeucl = true)

instances_maxvolume_all = [
    maxvolume1,
    maxvolume2,
    maxvolume3,
    maxvolume5,
    maxvolume6,
    maxvolume7,
    maxvolume8,
    maxvolume9,
    ]
instances_maxvolume_few = [
    maxvolume1,
    maxvolume2,
    maxvolume3,
    ]

function test_maxvolume(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return
end
