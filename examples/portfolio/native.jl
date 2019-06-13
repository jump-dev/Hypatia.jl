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
    G2 = vcat(zeros(1, num_stocks), -sigma_half)
    G = vcat(G1, G2)
    h = zeros(2 * num_stocks + 1)
    h[num_stocks + 1] = gamma
    cone_idxs = [1:num_stocks, (num_stocks + 1):(2 * num_stocks + 1)]
    cones = [CO.Nonnegative{Float64}(num_stocks), CO.EpiNormEucl{Float64}(num_stocks + 1)]

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

portfolio1() = portfolio(
    3,
    returns = -[0.0254,	0.0190,	0.0045],
    sigma_half = cholesky([
        0.0056	0.0012	0.0001
        0.0012	0.0020	0.0002
        0.0001	0.0002	0.0019
        ]).U,
    gamma = 0.033,
    )

test_portfolio_all(; options...) = test_portfolio.([
    portfolio1,
    ], options = options)

test_portfolio(; options...) = test_portfolio.([
    portfolio1,
    ], options = options)
