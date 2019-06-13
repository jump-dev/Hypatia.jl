#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

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
    gamma_sqr::Real = -1.0,
    )

    if isempty(returns)
        returns = randn(num_stocks)
    end
    if isempty(sigma_half)
        sigma_half = randn(num_stocks, num_stocks)
    end
    if gamma_sqr < 0
        gamma_sqr = rand()
    end

    c = returns
    A = ones(1, num_stocks)
    b = [1.0]
    G1 = -Matrix{Float64}(I, num_stocks, num_stocks)
    G2 = vcat(zeros(1, num_stocks), -sigma_half)
    G = vcat(G1, G2)
    h = zeros(2 * num_stocks + 1)
    h[num_stocks + 1] = sqrt(gamma_sqr)
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
    @show r.x
    @test r.status == :Optimal
    return
end

# example from DMD homework
portfolio1() = portfolio(
    3,
    returns = -[0.0254,	0.0190,	0.0045],
    sigma_half = cholesky([
        0.0056	0.0012	0.0001;
        0.0012	0.0020	0.0002;
        0.0001	0.0002	0.0019;
        ]).U,
    gamma_sqr = 0.014,
    )

test_portfolio_all(; options...) = test_portfolio.([
    portfolio1,
    ], options = options)

test_portfolio(; options...) = test_portfolio.([
    portfolio1,
    ], options = options)

;
