#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

maximize expected returns subject to risk constraints
=#

using LinearAlgebra
import Random
using Test
import Hypatia
import Hypatia.HypReal
const CO = Hypatia.Cones

function portfolio(
    T::Type{<:HypReal},
    num_stocks::Int,
    risk_measures::Vector{Symbol};
    use_l1ball::Bool = true,
    use_linfball::Bool = true,
    )
    if :entropic in risk_measures && length(risk_measures) > 1
        error("if using entropic ball, cannot specify other risk measures")
    end

    returns = rand(T, num_stocks)
    sigma_half = T.(randn(num_stocks, num_stocks))
    x = T.(randn(num_stocks))
    x ./= norm(x)
    gamma = sum(abs, sigma_half * x) / sqrt(T(num_stocks))

    c = returns
    # investments add to one
    A = ones(T, 1, num_stocks)
    b = T[1]
    # nonnegativity
    G = Matrix{T}(-I, num_stocks, num_stocks)
    h = zeros(T, num_stocks)
    cone_idxs = UnitRange{Int}[1:num_stocks]
    cones = CO.Cone{T}[CO.Nonnegative{T}(num_stocks)]
    cone_offset = num_stocks

    function add_single_ball(cone, gamma_new)
        G_risk = vcat(zeros(T, 1, num_stocks), -sigma_half)
        h_risk = vcat(gamma_new, zeros(T, num_stocks))
        G = vcat(G, G_risk)
        h = vcat(h, h_risk)
        push!(cones, cone)
        push!(cone_idxs, (cone_offset + 1):(cone_offset + num_stocks + 1))
        cone_offset += num_stocks + 1
    end

    if :quadratic in risk_measures
        add_single_ball(CO.EpiNormEucl{T}(num_stocks + 1), gamma)
    end
    if :l1 in risk_measures && use_l1ball
        add_single_ball(CO.EpiNormInf{T}(num_stocks + 1, true), gamma * sqrt(T(num_stocks)))
    end
    if :linf in risk_measures && use_linfball
        add_single_ball(CO.EpiNormInf{T}(num_stocks + 1), gamma)
    end

    if :l1 in risk_measures && !use_l1ball
        c = vcat(c, zeros(T, 2 * num_stocks))
        id = Matrix{T}(I, num_stocks, num_stocks)
        id2 = Matrix{T}(I, 2 * num_stocks, 2 * num_stocks)
        A_slacks = hcat(sigma_half, -id, id)
        A_l1 = hcat(zeros(T, 1, num_stocks), ones(T, 1, 2 * num_stocks))
        A = [
            A    zeros(T, 1, 2 * num_stocks);
            A_slacks;
            A_l1;
            ]
        b = vcat(b, zeros(T, num_stocks), gamma * sqrt(T(num_stocks)))
        G = [
            G    zeros(T, size(G, 1), 2 * num_stocks);
            zeros(T, 2 * num_stocks, num_stocks)    -id2;
            ]
        h = vcat(h, zeros(T, 2 * num_stocks))
        push!(cones, CO.Nonnegative{T}(2 * num_stocks))
        push!(cone_idxs, (cone_offset + 1):(cone_offset + 2 * num_stocks))
        cone_offset += 2 * num_stocks
    end

    if :linf in risk_measures && !use_linfball
        c = vcat(c, zeros(T, 2 * num_stocks))
        id = Matrix{T}(I, num_stocks, num_stocks)
        fill_cols = size(A, 2) - num_stocks
        fill_rows = size(A, 1)
        A = [
            A    zeros(T, fill_rows, 2 * num_stocks);
            sigma_half    zeros(T, num_stocks, fill_cols)    id    zeros(T, num_stocks, num_stocks);
            -sigma_half    zeros(T, num_stocks, fill_cols)    zeros(T, num_stocks, num_stocks)    id;
            ]
        b = vcat(b, gamma * ones(T, 2 * num_stocks))
        G = [
            G    zeros(T, size(G, 1), 2 * num_stocks);
            zeros(T, 2 * num_stocks, size(G, 2))    Matrix{T}(-I, 2 * num_stocks, 2 * num_stocks);
            ]
        h = vcat(h, zeros(T, 2 * num_stocks))
        push!(cones, CO.Nonnegative{T}(2 * num_stocks))
        push!(cone_idxs, (cone_offset + 1):(cone_offset + 2 * num_stocks))
        cone_offset += 2 * num_stocks
    end

    if :entropic in risk_measures
        # sigma_half = abs.(sigma_half) TODO will this always be feasible?
        c = vcat(c, zeros(T, 2 * num_stocks))
        A = [
            A    zeros(T, size(A, 1), 2 * num_stocks);
            zeros(T, 1, num_stocks)    ones(T, 1, 2 * num_stocks);
            ]
        b = vcat(b, gamma^2)
        G2pos = zeros(T, 3 * num_stocks, 2 * num_stocks + size(G, 2))
        G2neg = zeros(T, 3 * num_stocks, 2 * num_stocks + size(G, 2))
        h2 = zeros(T, 3 * num_stocks)

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
        G = [
            G    zeros(T, size(G, 1), 2 * num_stocks);
            G2pos;
            G2neg;
            ]
        h = vcat(h, h2, h2)
        for i in 1:(2 * num_stocks)
            push!(cone_idxs, (3 * (i - 1) + cone_offset + 1):(3 * i + cone_offset))
            push!(cones, CO.HypoPerLog{T}())
        end
        cone_offset += 6 * num_stocks
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

portfolio1(T::Type{<:HypReal}) = portfolio(T, 4, [:l1], use_l1ball = true)
portfolio2(T::Type{<:HypReal}) = portfolio(T, 6, [:l1], use_l1ball = false)
portfolio3(T::Type{<:HypReal}) = portfolio(T, 4, [:linf], use_linfball = true)
portfolio4(T::Type{<:HypReal}) = portfolio(T, 6, [:linf], use_linfball = false)
portfolio5(T::Type{<:HypReal}) = portfolio(T, 4, [:linf, :l1], use_linfball = true, use_l1ball = true)
portfolio6(T::Type{<:HypReal}) = portfolio(T, 6, [:linf, :l1], use_linfball = false, use_l1ball = false)

instances_portfolio_all = [
    portfolio1,
    portfolio2,
    portfolio3,
    portfolio4,
    portfolio5,
    portfolio6,
    ]
instances_portfolio_few = [
    portfolio1,
    portfolio3,
    portfolio5,
    portfolio6,
    ]

function test_portfolio(instance::Function; T::Type{<:HypReal} = Float64, test_options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs; test_options..., atol = tol, rtol = tol)
    @test r.status == :Optimal
    return
end
