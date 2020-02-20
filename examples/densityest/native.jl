#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

given a sequence of observations X_1,...,X_n with each Xᵢ in Rᵈ,
find a density function f maximizing the log likelihood of the observations
    min -∑ᵢ zᵢ
    -zᵢ + log(f(Xᵢ)) ≥ 0 ∀ i = 1,...,n
    ∫f = 1
    f ≥ 0

TODO maximize geomean objective should be more efficient
==#

using LinearAlgebra
import Random
using Test
import Hypatia
import Hypatia.BlockMatrix
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

include(joinpath(@__DIR__, "data.jl"))

function densityest(
    T::Type{<:Real},
    X::Matrix{<:Real},
    deg::Int;
    use_wsos::Bool = true,
    hypogeomean_obj::Bool = true, # use root of likelihood, else log of likelihood
    use_hypogeomean::Bool = true,
    sample_factor::Int = 100,
    )
    (num_obs, dim) = size(X)
    X = convert(Matrix{T}, X)

    domain = MU.Box{T}(-ones(T, dim), ones(T, dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, Ps, w) = MU.interpolate(domain, halfdeg, sample = true, calc_w = true, sample_factor = sample_factor)
    lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)
    basis_evals = Matrix{T}(undef, num_obs, U)
    for i in 1:num_obs, j in 1:U
        basis_evals[i, j] = lagrange_polys[j](X[i, :])
    end

    cones = CO.Cone{T}[]

    num_psd_vars = 0
    if use_wsos
        # U variables
        h_poly = zeros(T, U)
        b_poly = T[]
        push!(cones, CO.WSOSInterpNonnegative{T, T}(U, Ps))
    else
        # U polynomial coefficient variables plus PSD variables
        # there are length(Ps) new PSD variables, we will store them scaled, lower triangle, row-wise
        psd_var_list = Matrix{T}[]
        for i in eachindex(Ps)
            L = size(Ps[i], 2)
            dim = CO.svec_length(L)
            num_psd_vars += dim
            push!(psd_var_list, zeros(T, U, dim))
            idx = 1
            # we will work with PSD vars with scaled off-diagonals
            # relevant columns (not rows) in A need to be scaled by sqrt(2) also
            for k in 1:L
                for l in 1:(k - 1)
                    psd_var_list[i][:, idx] = Ps[i][:, k] .* Ps[i][:, l] * sqrt(T(2))
                    idx += 1
                end
                psd_var_list[i][:, idx] = Ps[i][:, k] .* Ps[i][:, k]
                idx += 1
            end
            push!(cones, CO.PosSemidefTri{T, T}(dim))
        end
        A_psd = hcat(psd_var_list...)
        b_poly = zeros(T, U)
        h_poly = zeros(T, num_psd_vars)
    end

    if hypogeomean_obj
        num_hypo_vars = 1
        if use_hypogeomean
            G_likl = [
                -one(T) zeros(T, 1, U)
                zeros(T, num_obs) -basis_evals
                ]
            h_likl = zeros(T, 1 + num_obs)
            push!(cones, CO.HypoGeomean{T}(fill(inv(T(num_obs)), num_obs)))
            A_ext = zeros(T, 0, num_obs)
        else
            num_ext_geom_vars = 1 + num_obs
            h_likl = zeros(T,  3 * num_obs + 2)
            # order of variables is: hypograph vars, f(obs), psd_vars, geomean ext vars (y, z)
            G_likl = zeros(T, 3 * num_obs + 2, 2 + U + num_psd_vars + num_obs)
            # u - y <= 0
            G_likl[1, :] .= vcat(one(T), zeros(T, U + num_psd_vars), -one(T), zeros(T, num_obs))
            push!(cones, CO.Nonnegative{T}(1))
            # e'z >= 0
            G_likl[2, :] .= vcat(zeros(T, 2 + U + num_psd_vars), -ones(T, num_obs))
            push!(cones, CO.Nonnegative{T}(1))
            # f(x) <= y * log(z / y)
            row_offset = 3
            # number of columns before extended variables start
            ext_offset = 2 + U + num_psd_vars
            for i in 1:num_obs
                G_likl[row_offset, ext_offset + i] = -1
                G_likl[row_offset + 1, ext_offset] = -1
                G_likl[row_offset + 2, 2:(1 + U)] = -basis_evals[i, :]
                row_offset += 3
                push!(cones, CO.HypoPerLog{T}(3))
            end
        end
    else
        num_hypo_vars = num_obs
        h_likl = zeros(T, 3 * num_obs)
        G_likl = zeros(T, 3 * num_obs, num_obs + U)
        offset = 1
        for i in 1:num_obs
            G_likl[offset + 2, (num_obs + 1):(num_obs + U)] = -basis_evals[i, :]
            h_likl[offset + 1] = 1
            G_likl[offset, i] = -1
            offset += 3
            push!(cones, CO.HypoPerLog{T}(3))
        end
    end

    # extended formulation variables for geomean come after psd ones, so psd vars were already accounted for in hypogeomean_obj && !use_hypogeomean path
    if !hypogeomean_obj || use_hypogeomean
        G_likl = hcat(G_likl, zeros(T, size(G_likl, 1), num_psd_vars))
        num_ext_geom_vars = 0
    end
    c = vcat(-ones(T, num_hypo_vars), zeros(T, U + num_psd_vars + num_ext_geom_vars))
    h = vcat(h_poly, h_likl)
    b = vcat(b_poly, one(T))

    if use_wsos
        A = hcat(zeros(T, 1, num_hypo_vars), w', zeros(T, 1, num_ext_geom_vars))
        G = [
            zeros(T, U, num_hypo_vars)  Matrix{T}(-I, U, U)  zeros(T, U, num_ext_geom_vars);
            G_likl;
            ]
    else
        A = [
            zeros(T, U, num_hypo_vars)    Matrix{T}(-I, U, U)    A_psd  zeros(T, U, num_ext_geom_vars);
            zeros(T, 1, num_hypo_vars)    w'    zeros(T, 1, num_psd_vars)  zeros(T, 1, num_ext_geom_vars);
            ]
        G = [
            zeros(T, num_psd_vars, num_hypo_vars + U)  -Matrix{T}(I, num_psd_vars, num_psd_vars)   zeros(T, num_psd_vars, num_ext_geom_vars);
            G_likl;
            ]
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

densityest(T::Type{<:Real}, num_obs::Int, n::Int, deg::Int; options...) = densityest(T, randn(T, num_obs, n), deg; options...)

densityest1(T::Type{<:Real}) = densityest(T, iris_data(), 4)
densityest2(T::Type{<:Real}) = densityest(T, iris_data(), 4, use_wsos = false)
densityest3(T::Type{<:Real}) = densityest(T, cancer_data(), 4)
densityest4(T::Type{<:Real}) = densityest(T, cancer_data(), 4, use_wsos = false)
densityest5(T::Type{<:Real}) = densityest(T, 50, 1, 4)
densityest6(T::Type{<:Real}) = densityest(T, 50, 1, 4, use_wsos = false)
densityest7(T::Type{<:Real}) = densityest(T, 20, 2, 4)
densityest8(T::Type{<:Real}) = densityest(T, 20, 2, 4, use_wsos = false)
densityest9(T::Type{<:Real}) = densityest(T, 50, 1, 4, use_hypogeomean = false)
densityest10(T::Type{<:Real}) = densityest(T, 50, 1, 4, use_wsos = false, use_hypogeomean = false)
densityest11(T::Type{<:Real}) = densityest(T, 20, 2, 4, use_hypogeomean = false)
densityest12(T::Type{<:Real}) = densityest(T, 20, 2, 4, use_wsos = false, use_hypogeomean = false)
densityest13(T::Type{<:Real}) = densityest(T, 50, 1, 4, hypogeomean_obj = false)
densityest14(T::Type{<:Real}) = densityest(T, 50, 1, 4, use_wsos = false, hypogeomean_obj = false)
densityest15(T::Type{<:Real}) = densityest(T, 20, 2, 4, hypogeomean_obj = false)
densityest16(T::Type{<:Real}) = densityest(T, 20, 2, 4, use_wsos = false, hypogeomean_obj = false)

instances_densityest_all = [
    densityest1,
    densityest2,
    densityest3,
    densityest4,
    densityest5,
    densityest6,
    densityest7,
    densityest8,
    densityest9,
    densityest10,
    densityest11,
    densityest12,
    densityest13,
    densityest14,
    densityest15,
    densityest16,
    ]
instances_densityest_few = [
    densityest1,
    densityest3,
    densityest5,
    densityest6,
    ]

function test_densityest(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return
end
