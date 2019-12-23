#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

given a sequence of observations X_1,...,X_n with each Xᵢ in Rᵈ,
find a density function f maximizing the log likelihood of the observations
    min -∑ᵢ zᵢ
    -zᵢ + log(f(Xᵢ)) ≥ 0 ∀ i = 1,...,n
    ∫f = 1
    f ≥ 0
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
    sample_factor::Int = 100,
    use_linops::Bool = false,
    )
    (nobs, dim) = size(X)
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
    basis_evals = Matrix{T}(undef, nobs, U)
    for i in 1:nobs, j in 1:U
        basis_evals[i, j] = lagrange_polys[j](X[i, :])
    end

    cones = CO.Cone{T}[]
    cone_offset = 1

    num_psd_vars = 0
    if use_wsos
        # U variables
        h_poly = zeros(T, U)
        b_poly = T[]
        push!(cones, CO.WSOSInterpNonnegative{T, T}(U, Ps))
        cone_offset += U
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
            cone_offset += dim
        end
        A_psd = hcat(psd_var_list...)
        b_poly = zeros(T, U)
        h_poly = zeros(T, num_psd_vars)
    end

    h_exp = zeros(T, 3 * nobs)
    G_exp = zeros(T, 3 * nobs, nobs + U)
    offset = 1
    for i in 1:nobs
        G_exp[offset, (nobs + 1):(nobs + U)] = -basis_evals[i, :]
        h_exp[offset + 1] = 1
        G_exp[offset + 2, i] = -1
        offset += 3
        push!(cones, CO.EpiPerExp{T}())
        cone_offset += 3
    end
    num_epi_vars = nobs

    (log_rows, log_cols) = size(G_exp)
    if use_linops
        if use_wsos
            A = BlockMatrix{T}(1,num_epi_vars + U,
                [w'],
                [1:1],
                [(num_epi_vars + 1):(num_epi_vars + U)]
                )
            G = BlockMatrix{T}(U + log_rows, num_epi_vars + U,
                [-I, G_exp],
                [1:U, (U + 1):(U + log_rows)],
                [(num_epi_vars + 1):(num_epi_vars + U), 1:(num_epi_vars + U)],
                )
        else
            A = BlockMatrix{T}(U + 1, num_epi_vars + U + num_psd_vars,
                [-I, A_psd, w'],
                [1:U, 1:U, (U + 1):(U + 1)],
                [(num_epi_vars + 1):(num_epi_vars + U), (num_epi_vars + U + 1):(num_epi_vars + U + num_psd_vars), (num_epi_vars + 1):(num_epi_vars + U)],
                )
            G = BlockMatrix{T}(num_psd_vars + log_rows, num_epi_vars + U + num_psd_vars,
                [-I, G_exp],
                [1:num_psd_vars, (num_psd_vars + 1):(num_psd_vars + log_rows)],
                [(num_epi_vars + U + 1):(num_epi_vars + U + num_psd_vars), 1:(num_epi_vars + U)],
                )
        end
    else
        if use_wsos
            A = hcat(zeros(T, 1, num_epi_vars), w')
            G = [
                zeros(T, U, num_epi_vars)    Matrix{T}(-I, U, U);
                G_exp;
                ]
        else
            A = [
                zeros(T, U, num_epi_vars)    Matrix{T}(-I, U, U)    A_psd;
                zeros(T, 1, num_epi_vars)    w'    zeros(T, 1, num_psd_vars);
                ]
            G = [
                zeros(T, num_psd_vars, num_epi_vars + U)  -Matrix{T}(I, num_psd_vars, num_psd_vars);
                G_exp   zeros(T, log_rows, num_psd_vars);
                ]
        end
    end
    h = vcat(h_poly, h_exp)
    c = vcat(-ones(T, num_epi_vars), zeros(T, U + num_psd_vars))
    b = vcat(b_poly, one(T))

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

densityest(T::Type{<:Real}, nobs::Int, n::Int, deg::Int; options...) = densityest(T, randn(T, nobs, n), deg; options...)

densityest1(T::Type{<:Real}) = densityest(T, iris_data(), 4)
densityest2(T::Type{<:Real}) = densityest(T, iris_data(), 4, use_wsos = false)
densityest3(T::Type{<:Real}) = densityest(T, cancer_data(), 4)
densityest4(T::Type{<:Real}) = densityest(T, cancer_data(), 4, use_wsos = false)
densityest5(T::Type{<:Real}) = densityest(T, 50, 1, 4)
densityest6(T::Type{<:Real}) = densityest(T, 50, 1, 4, use_wsos = false)
densityest7(T::Type{<:Real}) = densityest(T, 20, 2, 4)
densityest8(T::Type{<:Real}) = densityest(T, 20, 2, 4, use_wsos = false)
densityest9(T::Type{<:Real}) = densityest(T, 10, 1, 2, use_linops = true)
densityest10(T::Type{<:Real}) = densityest(T, 10, 1, 2, use_wsos = false, use_linops = true)

instances_densityest_all = [
    densityest1,
    densityest2,
    densityest3,
    densityest4,
    densityest5,
    densityest6,
    densityest7,
    densityest8,
    ]
instances_densityest_linops = [
    densityest9,
    densityest10,
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
