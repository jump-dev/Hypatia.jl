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
import DynamicPolynomials
import Hypatia
const HYP = Hypatia
const MO = HYP.Models
const MU = HYP.ModelUtilities
const CO = HYP.Cones
const SO = HYP.Solvers

THR = Type{<: HYP.HypReal}

const rt2 = sqrt(2)

include(joinpath(@__DIR__, "data.jl"))

function densityest(
    X::AbstractMatrix{Float64},
    deg::Int;
    use_sumlog::Bool = false,
    use_wsos::Bool = true,
    sample_factor::Int = 100,
    T::THR = Float64,
    )
    (nobs, dim) = size(X)

    X = convert(AbstractMatrix{T}, X)
    domain = MU.Box(-ones(dim), ones(dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, P0, PWts, w) = MU.interpolate(domain, halfdeg, sample = true, calc_w = true, sample_factor = sample_factor)

    lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)
    basis_evals = Matrix{Float64}(undef, nobs, U)
    for i in 1:nobs, j in 1:U
        basis_evals[i, j] = lagrange_polys[j](X[i, :])
    end

    # TODO remove these when converting is figured out
    P0c = convert.(T, P0)
    PWtsc = Matrix{T}[]
    for pwt in PWts
        push!(PWtsc, convert.(T, pwt))
    end

    cones = CO.Cone[]
    cone_idxs = UnitRange{Int}[]
    cone_offset = 1
    if use_wsos
        # will set up for U variables
        G_poly = -Matrix{T}(I, U, U)
        h_poly = zeros(T, U)
        c_poly = zeros(T, U)
        A_poly = zeros(T, 0, U)
        b_poly = Float64[]
        push!(cones, CO.WSOSPolyInterp{T, T}(U, [P0c, PWtsc...]))
        push!(cone_idxs, 1:U)
        cone_offset += U
        A_lin = zeros(T, 1, U)
        num_psd_vars = 0
    else
        # will set up for U coefficient variables plus PSD variables
        # there are n new PSD variables, we will store them scaled, lower triangle, row-wise
        n = length(PWts) + 1
        num_psd_vars = 0
        a_poly = Matrix{T}[]
        L = size(P0c, 2)
        dim = div(L * (L + 1), 2)
        num_psd_vars += dim
        # first part of A
        push!(a_poly, zeros(T, U, dim))
        idx = 1
        for k in 1:L, l in 1:k
            # off diagonals are doubled, but already scaled by rt2
            a_poly[1][:, idx] = P0c[:, k] .* P0c[:, l] * (k == l ? 1 : rt2)
            idx += 1
        end
        push!(cones, CO.PosSemidef{T, T}(dim))
        push!(cone_idxs, 1:dim)
        cone_offset += dim

        for i in 1:(n - 1)
            L = size(PWts[i], 2)
            dim = div(L * (L + 1), 2)
            num_psd_vars += dim
            push!(a_poly, zeros(T, U, dim))
            idx = 1
            for k in 1:L, l in 1:k
                # off diagonals are doubled, but already scaled by rt2
                a_poly[i + 1][:, idx] = PWtsc[i][:, k] .* PWtsc[i][:, l] * (k == l ? 1 : rt2)
                idx += 1
            end
            push!(cone_idxs, cone_offset:(cone_offset + dim - 1))
            push!(cones, CO.PosSemidef{T, T}(dim))
            cone_offset += dim
        end
        A_lin = zeros(T, 1, U + num_psd_vars)
        A_poly = hcat(a_poly...)
        A_poly = [-Matrix{T}(I, U, U) A_poly]
        b_poly = zeros(T, U)
        G_poly = [zeros(T, num_psd_vars, U) -Matrix{T}(I, num_psd_vars, num_psd_vars)]
        h_poly = zeros(T, num_psd_vars)
    end
    A_lin[1:U] = w

    if use_sumlog
        # pre-pad with one hypograph variable
        c_log = T[-1]
        G_poly = [zeros(T, cone_offset - 1, 1) G_poly]
        A = [
            zeros(T, size(A_poly, 1)) A_poly
            0 A_lin
            ]
        h_log = zeros(T, nobs + 2)
        h_log[2] = 1
        G_log = zeros(T, 2 + nobs, 1 + U + num_psd_vars)
        G_log[1, 1] = -1
        for i in 1:nobs
            G_log[i + 2, 2:(1 + U)] = -basis_evals[i, :]
        end
        push!(cone_idxs, cone_offset:(cone_offset + 1 + nobs))
        push!(cones, CO.HypoPerSumLog{T}(nobs + 2))
    else
        # pre-pad with `nobs` hypograph variables
        c_log = -ones(nobs)
        G_poly = [zeros(T, cone_offset - 1, nobs) G_poly]
        A = [
            zeros(T, size(A_poly, 1), nobs) A_poly
            zeros(T, 1, nobs) A_lin
            ]
        h_log = zeros(T, 3 * nobs)

        G_log = zeros(T, 3 * nobs, nobs + U + num_psd_vars)
        offset = 1
        for i in 1:nobs
            G_log[offset, i] = -1.0
            G_log[offset + 2, (nobs + 1):(nobs + U)] = -basis_evals[i, :]
            h_log[offset + 1] = 1.0
            offset += 3
            push!(cones, CO.HypoPerSumLog{T}(3))
            push!(cone_idxs, cone_offset:(cone_offset + 2))
            cone_offset += 3
        end
    end
    G = vcat(G_poly, G_log)
    h = vcat(h_poly, h_log)
    c = vcat(c_log, zeros(T, U), zeros(T, num_psd_vars))
    b = vcat(b_poly, 1)

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

densityest(nobs::Int, n::Int, deg::Int; options...) = densityest(randn(nobs, n), deg; options...)

densityest1(; T::THR = Float64) = densityest(iris_data(), 4, use_sumlog = true, T = T)
densityest2(; T::THR = Float64) = densityest(iris_data(), 4, use_sumlog = false, T = T)
densityest3(; T::THR = Float64) = densityest(cancer_data(), 4, use_sumlog = true, T = T)
densityest4(; T::THR = Float64) = densityest(cancer_data(), 4, use_sumlog = false, T = T)
densityest5(; T::THR = Float64) = densityest(200, 1, 4, use_sumlog = true, T = T)
densityest6(; T::THR = Float64) = densityest(200, 1, 4, use_sumlog = false, T = T)
densityest7(; T::THR = Float64) = densityest(200, 1, 4, use_sumlog = true, use_wsos = false, T = T)
densityest8(; T::THR = Float64) = densityest(200, 1, 4, use_sumlog = false, use_wsos = false, T = T)

function test_densityest(instance::Function; T::THR = Float64, rseed::Int = 1, options)
    Random.seed!(rseed)
    d = instance(T = T)
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

test_densityest_all(; T::THR = Float64, options...) = test_densityest.([
    densityest1,
    densityest2,
    densityest3,
    densityest4,
    densityest5,
    densityest6,
    densityest7,
    densityest8,
    ], T = T, options = options)

test_densityest(; T::THR = Float64, options...) = test_densityest.([
    densityest1,
    densityest3,
    densityest5,
    densityest6,
    densityest7,
    densityest8,
    ], T = T, options = options)
