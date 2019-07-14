#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see https://www.cvxpy.org/examples/dgp/pf_matrix_completion.html
modified to use spectral norm
=#

using LinearAlgebra
import Random
using Test
import Hypatia
import Hypatia.HypReal
const CO = Hypatia.Cones

function matrixcompletion(
    T::Type{<:HypReal},
    m::Int,
    n::Int;
    use_geomean::Bool = true,
    use_epinorm::Bool = true,
    )
    @assert m <= n
    rt2 = sqrt(T(2))

    num_known = round(Int, m * n * 0.1)
    known_rows = rand(1:m, num_known)
    known_cols = rand(1:n, num_known)
    known_vals = rand(T, num_known) .- T(0.5)

    mat_to_vec_idx(i::Int, j::Int) = (j - 1) * m + i

    is_known = fill(false, m * n)
    # h for the rows that X (the matrix and not epigraph variable) participates in
    h_norm_x = zeros(T, m * n)
    for (k, (i, j)) in enumerate(zip(known_rows, known_cols))
        known_idx = mat_to_vec_idx(i, j)
        # if not using the epinorminf cone, indices relate to X'
        h_norm_x[known_idx] = known_vals[k]
        is_known[known_idx] = true
    end

    num_known = sum(is_known) # if randomly generated, some indices may repeat
    num_unknown = m * n - num_known
    c = vcat(one(T), zeros(T, num_unknown))
    b = T[]

    # epinormspectral cone- get vec(X) in G and h
    if use_epinorm
        G_norm = zeros(T, m * n, num_unknown)
        total_idx = 1
        unknown_idx = 1
        for j in 1:n, i in 1:m
            if !is_known[total_idx]
                G_norm[total_idx, unknown_idx] = -1
                unknown_idx += 1
            end
            total_idx += 1
        end

        # add first row and column for epigraph variable
        G_norm = [
            -one(T)    zeros(T, 1, num_unknown);
            zeros(T, m * n)    G_norm;
            ]
        h_norm_x = vcat(zero(T), h_norm_x)
        h_norm = h_norm_x

        cones = CO.Cone{T}[CO.EpiNormSpectral{T}(m, n)]
        cone_idxs = UnitRange{Int}[1:(m * n + 1)]
        cone_offset = m * n + 1
    else
        num_rows = div(m * (m + 1), 2) + m * n + div(n * (n + 1), 2)
        G_norm = zeros(T, num_rows, num_unknown + 1)
        h_norm = zeros(T, num_rows)
        # first block epigraph variable * I
        for i in 1:m
            G_norm[sum(1:i), 1] = -1
        end
        offset = div(m * (m + 1), 2)
        # index to count rows in the bottom half of the large to-be-PSD matrix
        idx = 1
        # index only in X
        var_idx = 1
        # index of unknown vars (the x variables in the standard from), can increment it because we are moving row wise in X'
        unknown_idx = 1
        # fill bottom `n` rows
        for i in 1:n
            # X'
            for j in 1:m
                if !is_known[var_idx]
                    G_norm[offset + idx, 1 + unknown_idx] = -rt2
                    unknown_idx += 1
                else
                    h_norm[offset + idx] = h_norm_x[var_idx] * rt2
                end
                idx += 1
                var_idx += 1
            end
            # second block epigraph variable * I
            # skip `i` rows which will be filled with zeros
            idx += i
            G_norm[offset + idx - 1, 1] = -1
        end
        cones = CO.Cone{T}[CO.PosSemidef{T, T}(num_rows)]
        cone_idxs = UnitRange{Int}[1:num_rows]
        cone_offset = num_rows
    end

    if use_geomean
        # hypogeomean for values to be filled
        G_geo = zeros(T, num_unknown + 1, num_unknown + 1)
        total_idx = 1
        unknown_idx = 1
        for j in 1:n, i in 1:m
            if !is_known[mat_to_vec_idx(i, j)]
                G_geo[unknown_idx + 1, unknown_idx + 1] = -1
                unknown_idx += 1
            end
            total_idx += 1
        end
        # first component of the vector in the in geomean cone, elements multiply to one
        h2 = vcat(one(T), zeros(T, num_unknown))
        h = vcat(h_norm, h2)
        @assert total_idx - 1 == m * n
        @assert unknown_idx - 1 == num_unknown

        A = zeros(T, 0, 1 + num_unknown)
        push!(cone_idxs, (cone_offset + 1):(cone_offset + num_unknown + 1))
        push!(cones, CO.HypoGeomean{T}(fill(inv(T(num_unknown)), num_unknown)))
    else
        # number of 3-dimensional power cones needed is num_unknown - 1, number of new variables is num_unknown - 2
        # first num_unknown columns overlap with G_norm, column for the epigraph variable of the spectral cone added later
        G_geo = zeros(T, 3 * (num_unknown - 1), 2 * num_unknown - 2)
        # first cone is a special case since two of the original variables participate in it
        G_geo[3, 1] = -1
        G_geo[2, 2] = -1
        G_geo[1, num_unknown + 1] = -1
        push!(cones, CO.HypoGeomean{T}(fill(inv(T(2)), 2)))
        push!(cone_idxs, (cone_offset + 1):(cone_offset + 3))
        offset = 4
        # loop over new vars
        for i in 1:(num_unknown - 3)
            G_geo[offset, num_unknown + i + 1] = -1
            G_geo[offset + 1, num_unknown + i] = -1
            G_geo[offset + 2, i + 2] = -1
            push!(cones, CO.HypoGeomean{T}([T(i + 1) / T(i + 2), inv(T(i + 2))]))
            push!(cone_idxs, (cone_offset + 3 * i + 1):(cone_offset + 3 * (i + 1)))
            offset += 3
        end

        # last row also special becuase hypograph variable is fixed
        G_geo[offset + 2, num_unknown] = -1
        G_geo[offset + 1, 2 * num_unknown - 2] = -1
        push!(cones, CO.HypoGeomean{T}([T(num_unknown - 1) / T(num_unknown), inv(T(num_unknown))]))
        push!(cone_idxs, (cone_offset + 3 * num_unknown - 5):(cone_offset + 3 * num_unknown - 3))
        h = vcat(h_norm, zeros(T, 3 * (num_unknown - 2)), T[1, 0, 0])

        # G_norm needs to be post-padded with columns for 3dim cone vars
        G_norm = hcat(G_norm, zeros(T, size(G_norm, 1), num_unknown - 2))
        # G_geo needs to be pre-padded with the epigraph variable for the spectral norm cone
        G_geo = hcat(zeros(T, 3 * (num_unknown - 1)), G_geo)
        c = vcat(c, zeros(T, num_unknown - 2))
        A = zeros(T, 0, size(G_geo, 2))
    end
    G = vcat(G_norm, G_geo)

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

# matrixcompletion1(T::Type{<:HypReal}) = matrixcompletion(T, 5, 6)
# matrixcompletion2(T::Type{<:HypReal}) = matrixcompletion(T, 5, 6, use_geomean = false)
# matrixcompletion3(T::Type{<:HypReal}) = matrixcompletion(T, 5, 6, use_epinorm = false)
# matrixcompletion4(T::Type{<:HypReal}) = matrixcompletion(T, 5, 6, use_geomean = false, use_epinorm = false)
# matrixcompletion5(T::Type{<:HypReal}) = matrixcompletion(T, 6, 8)
# matrixcompletion6(T::Type{<:HypReal}) = matrixcompletion(T, 6, 8, use_geomean = false)
# matrixcompletion7(T::Type{<:HypReal}) = matrixcompletion(T, 6, 8, use_epinorm = false)
# matrixcompletion8(T::Type{<:HypReal}) = matrixcompletion(T, 6, 8, use_geomean = false, use_epinorm = false)
# matrixcompletion9(T::Type{<:HypReal}) = matrixcompletion(T, 8, 8)
# matrixcompletion10(T::Type{<:HypReal}) = matrixcompletion(T, 8, 8, use_geomean = false)
# matrixcompletion11(T::Type{<:HypReal}) = matrixcompletion(T, 8, 8, use_epinorm = false)
# matrixcompletion12(T::Type{<:HypReal}) = matrixcompletion(T, 8, 8, use_geomean = false, use_epinorm = false)
#
# instances_matrixcompletion_all = [
#     matrixcompletion1,
#     matrixcompletion2,
#     matrixcompletion3,
#     matrixcompletion4,
#     matrixcompletion5,
#     matrixcompletion6,
#     matrixcompletion7,
#     matrixcompletion8,
#     matrixcompletion9,
#     matrixcompletion10,
#     matrixcompletion11,
#     matrixcompletion12,
#     ]
# instances_matrixcompletion_few = [
#     matrixcompletion1,
#     matrixcompletion2,
#     matrixcompletion3,
#     matrixcompletion4,
#     ]
#
# function test_matrixcompletion(instance::Function; T::Type{<:HypReal} = Float64, test_options::NamedTuple = NamedTuple(), rseed::Int = 1)
#     Random.seed!(rseed)
#     tol = max(1e-5, sqrt(sqrt(eps(T))))
#     d = instance(T)
#     r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs; test_options..., atol = tol, rtol = tol)
#     @test r.status == :Optimal
#     return
# end



function runmc(T::Type{<:HypReal})
    Random.seed!(1)
    d = matrixcompletion(T, 5, 7)

    tol = eps(T)

    try
        r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs,
            solver_options = (
                verbose = true,
                tol_rel_opt = tol,
                tol_abs_opt = tol,
                tol_feas = tol,
                max_iters = 2000,
                ),
            test = false, atol = tol, rtol = tol)
    catch e
        println()
        println(e)
        println()
    end
    return
end


println("\ndo a compile run first!")

println("\nnon-BigFloat")
using Quadmath
# using DoubleFloats
for T in (Float32, Float64, Float128)
    println()
    @show T
    @show eps(T)
    @show sqrt(eps(T))
    println()
    @time runmc(T)
    println("\ndone\n")
end


println("\nBigFloat")
for P in [2^i for i in 5:11]
    println()
    @show P
    setprecision(P)
    T = BigFloat
    @show eps(T)
    @show sqrt(eps(T))
    println()
    @time runmc(T)
    println("\ndone\n")
end



#=
for 5, 7

T = Float32
eps(T) = 1.1920929f-7
sqrt(eps(T)) = 0.00034526698f0
1e-5
1e-5
42
0.528663 seconds (311.68 k allocations: 10.943 MiB, 3.42% gc time)

T = Float64
eps(T) = 2.220446049250313e-16
sqrt(eps(T)) = 1.4901161193847656e-8
1e-9
1e-14
86
1.329719 seconds (1.47 M allocations: 67.558 MiB, 2.21% gc time)

T = Float128
eps(T) = 1.92592994438723585305597794258492732e-34
sqrt(eps(T)) = 1.38777878078144567552953958511352539e-17
1e-17
1e-17
42
1.208103 seconds (985.27 k allocations: 50.008 MiB, 1.81% gc time)


BigFloat

P = 32
eps(T) = 4.6566128731e-10
sqrt(eps(T)) = 2.1579186438e-05
1e-6
1e-6
22
1.986621 seconds (23.12 M allocations: 892.201 MiB, 23.32% gc time)

P = 64
eps(T) = 1.08420217248550443401e-19
sqrt(eps(T)) = 3.29272253991359623327e-10
1e-11
1e-11
30
1.958444 seconds (29.86 M allocations: 1.114 GiB, 28.50% gc time)

P = 128
eps(T) = 5.877471754111437539843682686111228389093e-39
sqrt(eps(T)) = 7.666467083416870407194249863103337360462e-20
1e-20
1e-20
47
3.061280 seconds (45.90 M allocations: 2.053 GiB, 25.00% gc time)

P = 256
eps(T) = 1.727233711018888925077270372560079914223200072887256277004740694033718360632485e-77
sqrt(eps(T)) = 4.156000133564589919546802616566101790707151631732974255008686036787230203508115e-39
1e-39
1e-39
84
6.277647 seconds (80.87 M allocations: 4.217 GiB, 29.32% gc time)

P = 512
eps(T) = 1.49166814624004134865819306309258676747529430692008137885430366664125567701402366098723497808008556067230232065116722029068254561904506053209723296591841694e-154
sqrt(eps(T)) = 1.22133866975546195086019597071117283027094025981639387493045310083030448293378253341035004249805099426031227511728968498663477709879906332153767495400217661e-77
1e-77
1e-77
158
15.260882 seconds (150.70 M allocations: 10.094 GiB, 35.14% gc time)

P = 1024
eps(T) = 1.112536929253600691545116358666202032109607990231165915276663708443602217406959097927141579506255510282033669865517905502576217080776730054428006192688859410565388996766001165239805073721291818035960782523471251867104187625403325308329079474360245589984295819824250317954385059152437399890443876874974725790226e-308
sqrt(eps(T)) = 1.054768661486299891265269955921447617668122146951123818422873616346378127515077671186446342376160961141355568310236688317419448260822933627343237087908143366084050548700019311119589413204950077818402241353583185793507898633312913126888304085897455970251805951412361943485111158467502577654049156080510910528569e-154
1e-155
1e-155
306
41.478292 seconds (290.45 M allocations: 28.068 GiB, 33.16% gc time)

P = 2048
eps(T) = 6.18869209476515655096036673994239570778511260776993809190819691643404272938245996285270589311251959050648287585080362350439337317594971660034473673627746625037212214856928747398094197037459410910859856169137483106413173821435053654722982815983890299679351640343926950553743264157286076769809516794265257963240241151408522789563836196134401020208526110355392769698516103152645684438841021105643996049099009401123202572924582440320070494342603031609106945661480835379003673392053504811854029060965070501336226472619954975751335858537226311222169002208692675741211119426352381663086239740969262313789776354679555813722976e-617
sqrt(eps(T)) = 7.86682406995679327614617677464075897028061956978617697426774484830730478306559967284048304807053314278887298155371492424732633188955612074050773059835520663362136412703251238543706594502564416391860015405092345887187696875300998071599491250317708138163119987437740277289093498494958539811599774718332193630164402651368559504900152418822252762297740386072897269338256373452619301070238066368943114007884793981599376936208391249731263559518805526665165772208457878556378152467834780392834126284513006756561891004602211852513388264700428547460401974050604649319969579704454641379872051444425644592142153686733539402369319e-309
1e-304
1e-304
607
135.488751 seconds (596.23 M allocations: 92.311 GiB, 26.48% gc time)

=#
