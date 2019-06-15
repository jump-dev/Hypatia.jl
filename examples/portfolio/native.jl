#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

TODO
- add description
- add more risk constraints covering more cones
- enable random generation
=#

using LinearAlgebra
import Random
using Test
import DataFrames
import CSV
import DynamicPolynomials
import Hypatia
const HYP = Hypatia
const MO = HYP.Models
const MU = HYP.ModelUtilities
const CO = HYP.Cones
const SO = HYP.Solvers

function portfolio(
    num_stocks::Int;
    returns::Vector{Float64} = Float64[],
    sigma_half::AbstractMatrix{Float64} = Matrix{Float64}(undef, 0, 0),
    gamma::Float64 = -1.0,
    risk_measure::Symbol = :entropic,
    )

    if isempty(returns)
        returns = randn(num_stocks)
    end
    if isempty(sigma_half)
        sigma_half = randn(num_stocks, num_stocks)
    end
    if gamma < 0
        gamma = rand()
    end

    c = returns
    A = ones(1, num_stocks)
    b = [1.0]
    G1 = -Matrix{Float64}(I, num_stocks, num_stocks)
    h1 = zeros(num_stocks)
    cone_idxs = [1:num_stocks]
    cones = CO.Cone[CO.Nonnegative{Float64}(num_stocks)]

    if risk_measure == :quadratic
        G2 = vcat(zeros(1, num_stocks), -sigma_half)
        h2 = [gamma; zeros(num_stocks)]
        cone_idxs = vcat(cone_idxs, [(num_stocks + 1):(2 * num_stocks + 1)])
        cones = vcat(cones, [CO.EpiNormEucl{Float64}(num_stocks + 1)])
        G = vcat(G1, G2)
        h = vcat(h1, h2)
    elseif risk_measure == :entropic
        # sigma_half = abs.(sigma_half) TODO will this always be feasible?
        c = vcat(c, zeros(2 * num_stocks))
        A = [A zeros(1, 2 * num_stocks); zeros(1, num_stocks) ones(1, 2 * num_stocks)]
        b = vcat(b, gamma^2)
        G2pos = zeros(3 * num_stocks, 3 * num_stocks)
        G2neg = zeros(3 * num_stocks, 3 * num_stocks)
        h2 = zeros(3 * num_stocks)

        offset = 1
        for i in 1:num_stocks
            G2pos[offset, num_stocks + i] = 1 # entropy
            G2pos[offset + 1, 1:num_stocks] = -sigma_half[i, :]
            h2[offset + 1] = 1
            h2[offset + 2] = 1
            G2neg[offset, 2 * num_stocks + i] = 1 # entropy
            G2neg[offset + 1, 1:num_stocks] = sigma_half[i, :]
            offset += 3
        end
        G = [G1 zeros(num_stocks, 2 * num_stocks); G2pos; G2neg]
        h = vcat(h1, h2, h2)
        cone_idxs = vcat(cone_idxs, [(3 * (i - 1) + num_stocks + 1):(3 * i + num_stocks) for i in 1:(2 * num_stocks)])
        cones = vcat(cones, [CO.HypoPerLog{Float64}() for _ in 1:(2 * num_stocks)])
    else
        error("unknown risk measure: $risk_measure")
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

function test_portfolio(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    model = MO.PreprocessedLinearModel{Float64}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{Float64}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

portfolio_ex(risk_measure::Symbol) = portfolio(
    3,
    returns = -[0.0254,	0.0190,	0.0045],
    sigma_half = cholesky([
        0.0056	0.0012	0.0001
        0.0012	0.0020	0.0002
        0.0001	0.0002	0.0019
        ]).U,
    gamma = 0.033,
    risk_measure = risk_measure,
    )

portfolio1() = portfolio_ex(:quadratic)
portfolio2() = portfolio_ex(:entropic)

test_portfolio_all(; options...) = test_portfolio.([
    portfolio1,
    portfolio2,
    ], options = options)

test_portfolio(; options...) = test_portfolio.([
    portfolio1,
    portfolio2,
    ], options = options)
