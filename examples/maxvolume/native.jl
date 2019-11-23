#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhedron
=#

using LinearAlgebra
import Random
using Test
import Hypatia
const CO = Hypatia.Cones
# TODO remove
import MathOptInterface

function maxvolume(
    T::Type{<:Real},
    n::Int;
    constr_cone::Symbol = :soc,
    )

    poly_hrep = Matrix{T}(I, n, n)
    poly_hrep .+= randn(n, n) * T(10) ^ (-n)
    c = vcat(-1, zeros(T, n))
    A = hcat(zeros(T, n), poly_hrep)
    b = ones(T, n)

    if constr_cone == :geomean
        G = -Matrix{T}(I, n + 1, n + 1)
        h = zeros(T, n + 1)
        cones = CO.Cone{T}[CO.HypoGeomean{T}(fill(inv(T(n)), n))]

    elseif constr_cone == :power3d
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

    elseif constr_cone == :soc
        # number of variables inside geometric mean is n
        # number of layers of variables
        l = MathOptInterface.Bridges.Constraint.ilog2(n)
        @show l
        # number of new variables = 1 + 2 + ... + 2^(l - 1) = 2^l - 1
        N = 2 ^ l
        num_new_vars = N - 1

        c = vcat(c, zeros(T, num_new_vars))
        A = hcat(A, zeros(T, n, num_new_vars))
        rtN = sqrt(T(N))
        # excludes original hypograph variable, padded later
        G_rsoc = zeros(T, 3 * num_new_vars, n + num_new_vars)
        cones = CO.Cone{T}[]

        offset = offset_next = 0
        row = 1
        # loop over layers, layer 1 describes hypograph variable
        for i in 1:l
            incr = 2 ^ (i - 1)
            offset_next = offset + incr
            # loop over variables in each layer
            for j in 1:incr
                if i == l
                    # in the last layer, we use the original variables
                    if 2j - 1 > n
                        # if we are beyond the number of variables in the actual geometric mean, we are adding the buffer variable
                        G_rsoc[row, n + 1] = -inv(rtN)
                    else
                        G_rsoc[row, 2j - 1] = -1
                    end
                    if 2j > n
                        # if we are beyond the number of variables in the actual geometric mean, we are adding the buffer variable
                        G_rsoc[row + 1, n + 1] = -inv(rtN)
                    else
                        G_rsoc[row + 1, 2j] = -1
                    end
                else
                    G_rsoc[row, n + offset_next + 2j - 1] = -1
                    G_rsoc[row + 1, n + offset_next + 2j] = -1
                end
                G_rsoc[row + 2, n + offset + j] = -1
                push!(cones, CO.EpiPerSquare{T}(3))
                row += 3
            end
            offset = offset_next
        end
        # account for original hypograph variable
        G = [
            zeros(T, 3(N - 1))  G_rsoc;
            one(T)  zeros(T, 1, n)  -inv(rtN)  zeros(T, 1, num_new_vars - 1);
            zeros(T, num_new_vars, n + 1)  -Matrix{T}(I, num_new_vars, num_new_vars)
            ]
        push!(cones, CO.Nonnegative{T}(1))
        # TODO does this need to be imposed for all variables explicitly?
        push!(cones, CO.Nonnegative{T}(num_new_vars))
        h = zeros(T, 3 * (N - 1) + 1 + num_new_vars)

    else
        error("unknown cone $(constr_cone)")
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

maxvolume1(T::Type{<:Real}) = maxvolume(T, 10, constr_cone = :geomean)
maxvolume2(T::Type{<:Real}) = maxvolume(T, 10, constr_cone = :power3d)
maxvolume3(T::Type{<:Real}) = maxvolume(T, 10, constr_cone = :soc)

maxvolume_all = [
    maxvolume1,
    maxvolume2,
    maxvolume3,
    ]
maxvolume_few = [
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

test_maxvolume.(maxvolume_all)
